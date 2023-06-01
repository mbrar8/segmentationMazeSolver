import torch.nn as nn
import torch


class UNetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class UNetDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranpose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x, skip_x):
        x = self.upconv(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.down1 = UNetEncoder(3, 64)
        self.down2 = UNetEncoder(64, 128)
        self.down3 = UNetEncoder(128, 256)
        self.down4 = UNetEncoder(256, 512)

        # Decoder
        self.up1 = UNetDecoder(512, 256)
        self.up2 = UNetDecoder(256, 128)
        self.up3 = UNetDecoder(128, 64)

        # Output
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Output
        x = self.out(x)
        return x



