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

        self.p_conv = DoubleConv(input_ch, 2*s)

        self.down1 = Down(2*s, 4*s)
        self.down2 = Down(4*s, 8*s)
        self.down3 = Down(8*s, 16*s)
        self.down4 = Down(16*s, 16*s)
        self.up1 = Up(32*s, 8*s)
        self.up2 = Up(16*s, 4*s)
        self.up3 = Up(8*s, 2*s)
        self.up4 = Up(4*s, 2*s)

        self.s_conv = DoubleConv(2*s, s)
        self.f_conv = DoubleConv(3*s, s)
        
        self.outc = OutConv(2*s, output_ch)

    def forward(self, a, b, c):
        za = self.p_conv(a)
        zb = self.p_conv(b)
        zc = self.p_conv(c)
        
        xb2 = self.down1(zb)
        xb3 = self.down2(xb2)
        xb4 = self.down3(xb3)
        xb5 = self.down4(xb4)

        xb = self.up1(xb5, xb4)
        xb = self.up2(xb, xb3)
        xb = self.up3(xb, xb2)
        xb = self.up4(xb, zb)
        xb = self.s_conv(xb)

        yab = torch.cat([xb, za], dim=1)
        yab = self.f_conv(yab)

        ybc = torch.cat([xb, zc], dim=1)
        ybc = self.f_conv(ybc)

        out = torch.cat([yab, ybc], dim=1)
        out = self.outc(out)

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