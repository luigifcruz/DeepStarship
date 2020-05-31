import os
import shutil
from tqdm import tqdm
import numpy as np
import re

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
import torchvision.datasets as datasets

from dataset import ExternalInputIterator, ExternalSourcePipeline
from eval import eval_net
from test import test_net
from utils import safe_div, AverageMeter, get_lr

from nvidia.dali.plugin.pytorch import DALIGenericIterator

import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, "a")        
    handler.setFormatter(formatter)

    name = np.random.randint(2**32)
    logger = logging.getLogger(str(name))
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def load_model(model, save_dir):
    if os.path.isdir(save_dir):
        res = input(f"Directory already exists: {save_dir}\n1 - Use it anyway.\n2 - Delete.\n3 - Exit.\n>> ")
        if res == '2':
            shutil.rmtree(save_dir)
        if res == '3':
            exit(0)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Load Existing Models
    cks = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(save_dir)
           if re.match(r'(.*)\.(pth)', f)]
    cks.sort()
 
    if len(cks) > 0:
        epoch = cks[-1]
        latest = "model_save_epoch_{}.pth".format(epoch)

        print("Loading model Epoch:", epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, latest)))
    else:
        epoch = 0

    return model, epoch


def run(model, net_size, root_dir, save_dir, input_size, batch_size, learning_rate, min_lr, epochs, device, patience):
    model = model(input_ch=3, output_ch=3, net_size=net_size)

    model, epoch = load_model(model, save_dir)
    
    # Load Datasets
    elements = ['im1', 'im2', 'im3', 'targets']

    train_iter = ExternalInputIterator(batch_size=batch_size, data_dir=os.path.join(root_dir, 'train'))
    train_pipe = ExternalSourcePipeline(data_iterator=iter(train_iter), batch_size=batch_size, num_threads=8, device_id=0, size=input_size)
    train_pipe.build()
    train_dali_iter = DALIGenericIterator([train_pipe], elements, train_iter.n)

    valid_iter = ExternalInputIterator(batch_size=1, data_dir=os.path.join(root_dir, 'valid'))
    valid_pipe = ExternalSourcePipeline(data_iterator=iter(valid_iter), batch_size=1, num_threads=4, device_id=0, size=input_size, eval_enabled=True)
    valid_pipe.build()
    valid_dali_iter = DALIGenericIterator([valid_pipe], elements, valid_iter.n)
    
    test_iter = ExternalInputIterator(batch_size=1, data_dir=os.path.join(root_dir, 'test'))
    test_pipe = ExternalSourcePipeline(data_iterator=iter(test_iter), batch_size=1, num_threads=1, device_id=0, size=input_size, eval_enabled=True)
    test_pipe.build()
    test_dali_iter = DALIGenericIterator([test_pipe], elements, 1)

    # Build the Network
    model = model.to(device)

    global_step = epoch * train_iter.n
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=patience)
    criterion = torch.nn.MSELoss()

    # Logging Settings
    logger = setup_logger(os.path.join(save_dir, "model_run.log"))
    writer = SummaryWriter(log_dir=save_dir, purge_step=global_step)

    info = f'''
    Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning Rate:   {learning_rate}
    Minimum LR:      {min_lr}
    Patience:        {patience}
    Training size:   {train_iter.n}
    Validation size: {valid_iter.n}
    Save Directory:  {save_dir}
    Model Directory: {root_dir}
    Device:          {device.type}
    Network Size:    {net_size}
    '''
    logger.info(info)
    print(info)
    
    # Run the Model
    for epoch in range(epoch, epochs):
        model.train()

        train_loss = AverageMeter()
        
        with tqdm(total=train_iter.n, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for _, it in enumerate(train_dali_iter):
                batch_data = it[0]

                im1, im2, im3 = batch_data["im1"], batch_data["im2"], batch_data["im3"]
                targets = batch_data["targets"]

                output = model(im1, im2, im3)

                loss = criterion(output, targets)
                train_loss.update(loss.item(), im1.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Training Loss', loss.item(), global_step)

                pbar.set_postfix(**{'loss_l2': train_loss.avg})
                pbar.update(im1.shape[0])

                if (global_step % 256) == 0:
                    writer.add_images('training_inputs', im2[:1,:,:,:], global_step)
                    writer.add_images('training_output', output[:1,:,:,:], global_step)
                    writer.add_images('training_target', targets[:1,:,:,:], global_step)

                    i, o = test_net(model, test_dali_iter)
                    test_dali_iter.reset()

                    writer.add_images('test_input', i[:1,:,:,:], global_step)
                    writer.add_images('test_output', o[:1,:,:,:], global_step)

                global_step += 1

            val_loss = eval_net(model, valid_dali_iter, device, valid_iter.n, writer, global_step)
            scheduler.step(val_loss)

            if get_lr(optimizer) <= min_lr:
                logger.info('Minimum Learning Rate Reached: Early Stopping')
                break
                
            writer.add_scalar('Validation Loss', val_loss, global_step)
            writer.add_scalar('Learning Rate', get_lr(optimizer), global_step)

            torch.save(model.state_dict(), os.path.join(save_dir, "model_save_epoch_{}.pth".format(epoch)))
            logger.info('Checkpoint {} saved!'.format(epoch))
            logger.info('Validation Loss L2: {}'.format(val_loss))
            logger.info('Learning Rate: {}'.format(get_lr(optimizer)))

            train_dali_iter.reset()
            valid_dali_iter.reset()

    writer.close()
    logger.info('Training finished, exiting...')
    torch.save(model.state_dict(), os.path.join(save_dir, "model_save_epoch_{}.pth".format(epoch)))
    logger.info('Final checkpoint {} saved!'.format(epoch))
    del model
