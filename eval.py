from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import utils

def eval_net(net, loader, device, n_val, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot_l2 = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for _, it in enumerate(loader):
            batch_data = it[0]
            im1, im2, im3 = batch_data["im1"], batch_data["im2"], batch_data["im3"]
            targets = batch_data["targets"]

            output_pred = net(im1, im2, im3)
            
            for out, pred in zip(targets, output_pred):
                tot_l2 += F.mse_loss(pred, out).item()
                
            pbar.update(im1.shape[0])

    return tot_l2/n_val