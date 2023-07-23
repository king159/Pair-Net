import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTiny(nn.Module):
    """
    a tiny cnn, parameters 0.2M
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 7,
        mid_channels: int = 64,
        layers: int = 3,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=kernel_size, padding=3
                ),
                nn.ReLU(inplace=True),
            )
        )
        for _ in range(layers - 2):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        mid_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                        padding=3,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=kernel_size, padding=3
                )
            )
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        return x.squeeze(1)


class ConvSmall(nn.Module):
    """
    a convnext-like cnn, parameters 2M
    """

    def __init__(self, dim=96):
        super().__init__()
        self.in_conv = nn.Conv2d(1, dim, kernel_size=7, padding=3)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = nn.LayerNorm([dim, 100, 100], eps=1e-6)
        self.pwconv1 = nn.Conv2d(
            dim, 4 * dim, kernel_size=1
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.out_conv = nn.Conv2d(dim, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x = x.unsqueeze(1)
        input = x
        x = self.in_conv(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.out_conv(x)
        x = input + x
        return x.squeeze(1)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvBase(nn.Module):
    """
    a u-net like cnn, parameters 31M
    """

    def __init__(self, in_channels=1, out_channels=1, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x.squeeze(1)


def creat_cnn(name: str) -> nn.Module:
    assert name in ["conv_tiny", "conv_small", "conv_base"]
    if name == "conv_tiny":
        return ConvTiny()
    elif name == "conv_small":
        return ConvSmall()
    elif name == "conv_base":
        return ConvBase()
    else:
        raise NotImplementedError


# if __name__ == "__main__":
#     model = creat_cnn("convsmall")
#     input = torch.rand((2, 100, 100))
#     output = model(input)
#     print(output.shape)
#     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     print(sum([torch.numel(p) for p in model_parameters]))
