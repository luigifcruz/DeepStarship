from unet import Classic_UNet
from PIL import Image
import torch
import os
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from shutil import copyfile

def open_img(path):
    image = Image.open(path)
    image = image.resize((int(image.size[0]/1.25), int(image.size[1]/1.25)), Image.BICUBIC)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x

mp = 'runs/B5/model_save_epoch_33.pth'

for n in range(1, 400):
    im1 = open_img('/media/luigifcruz/HDD1/Science/apple_hd/1080_3/output_{:03d}.png'.format(n))
    im2 = open_img('/media/luigifcruz/HDD1/Science/apple_hd/1080_3/output_{:03d}.png'.format(n+1))
    im3 = open_img('/media/luigifcruz/HDD1/Science/apple_hd/1080_3/output_{:03d}.png'.format(n+2))

    model = Classic_UNet(input_ch=3, output_ch=3, net_size=16)
    model.load_state_dict(torch.load(mp))

    model.eval()

    outputs = model(im1, im2, im3)

    save_image(outputs, '/media/luigifcruz/HDD1/Science/apple_hd/1440/output_{:03d}.png'.format(n))

exit()

for n in range(1, 1898):
    im1 = '/media/luigifcruz/HDD1/Science/new_starship/output_{:03d}.jpg'.format(n)
    im2 = '/media/luigifcruz/HDD1/Science/new_starship/output_{:03d}.jpg'.format(n+1)
    im3 = '/media/luigifcruz/HDD1/Science/new_starship/output_{:03d}.jpg'.format(n+2)

    os.makedirs('/media/luigifcruz/HDD1/Science/new_starship/dataset/{:05d}'.format(n))

    copyfile(im1, '/media/luigifcruz/HDD1/Science/new_starship/dataset/{:05d}/im1.jpg'.format(n))
    copyfile(im2, '/media/luigifcruz/HDD1/Science/new_starship/dataset/{:05d}/im2.jpg'.format(n))
    copyfile(im3, '/media/luigifcruz/HDD1/Science/new_starship/dataset/{:05d}/im3.jpg'.format(n))
