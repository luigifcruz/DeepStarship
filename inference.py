from unet import Classic_UNet
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

def open_img(path):
    image = Image.open(path)
    image = image.resize((int(1920*1.5), int(1080*1.5)), Image.BICUBIC)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x

mp = 'runs/A2/model_save_epoch_24.pth'

for n in range(1, 243):
    im1 = open_img('/media/luigifcruz/HDD/Science/test/output_{:03d}.png'.format(n))
    im2 = open_img('/media/luigifcruz/HDD/Science/test/output_{:03d}.png'.format(n+1))
    im3 = open_img('/media/luigifcruz/HDD/Science/test/output_{:03d}.png'.format(n+2))

    model = Classic_UNet(input_ch=3, output_ch=3, net_size=6)
    model.load_state_dict(torch.load(mp))

    model.eval()

    outputs = model(im1, im2, im3)

    save_image(outputs, 'out/image{:03d}.png'.format(n))