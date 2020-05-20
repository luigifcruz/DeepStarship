""" Original code from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py """

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_colors as colors


class Classic_UNet(nn.Module):
    def __init__(self, input_ch, output_ch, net_size=1):
        super(Classic_UNet, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch

        s = net_size

        self.inc = DoubleConv(input_ch, 2*s)
        self.inc2 = DoubleConv(2*s*3, 2*s)

        self.down1 = Down(2*s, 4*s)
        self.down2 = Down(4*s, 8*s)
        self.down3 = Down(8*s, 16*s)
        self.down4 = Down(16*s, 16*s)
        self.up1 = Up(32*s, 8*s)
        self.up2 = Up(16*s, 4*s)
        self.up3 = Up(8*s, 2*s)
        self.up4 = Up(4*s, 2*s)
        self.outc = OutConv(2*s, output_ch)

    def forward(self, im1, im2, im3):

        x1 = torch.cat([
            self.inc(im1),
            self.inc(im2),
            self.inc(im3)
        ], dim=1)
        x1 = self.inc2(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)

        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        m = nn.Sigmoid()
        return m(x)