import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class UNetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )


    def forward(self, x):
        return self.encode(x)

class UNetDecoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.doubleconv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_x):
        x = self.upconv(x)

        dY = skip_x.size()[2] - x.size()[2]
        dX = skip_x.size()[3] - x.size()[3]

        x = F.pad(x, [dX // 2, dX - dX // 2,
                    dY // 2, dY - dY // 2])


        x = torch.cat([skip_x, x], dim=1)
        return self.doubleconv(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = UNetEncoder(64, 128)
        self.down2 = UNetEncoder(128, 256)
        self.down3 = UNetEncoder(256, 512)
        self.down4 = UNetEncoder(512, 1024)


        # Decoder
        self.up1 = UNetDecoder(1024, 512)
        self.up2 = UNetDecoder(512, 256)
        self.up3 = UNetDecoder(256, 128)
        self.up4 = UNetDecoder(128, 64)

        # Output
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        print(x1.shape)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        x = self.out(x)
        return x



