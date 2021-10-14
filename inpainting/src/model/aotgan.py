import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .common import BaseNetwork


class InpaintGenerator(BaseNetwork):
    def __init__(self, args=None):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=2,
                out_channels=64,
                kernel_size=(1, 7, 7),
                padding=(0, 3, 3),
                padding_mode="replicate",
            ),
            nn.ReLU(True),
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                padding_mode="replicate",
            ),
            nn.ReLU(True),
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                padding_mode="replicate",
            ),
            nn.ReLU(True),
        )

        if args is not None:
            self.middle = nn.Sequential(
                *[AOTBlock(256, args.rates) for _ in range(args.block_num)]
            )
        else:
            self.middle = nn.Sequential(
                *[AOTBlock(256, [1, 2, 4, 8]) for _ in range(8)]
            )

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv3d(
                in_channels=64,
                out_channels=1,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                padding_mode="replicate",
            ),
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=2)
        x = torch.transpose(x, 1, 2)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.transpose(x, 1, 2)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = nn.Upsample(
            scale_factor=(1, scale, scale), mode="trilinear", align_corners=True
        )
        self.conv = nn.Conv3d(
            in_channels=inc,
            out_channels=outc,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )

        self.layer = nn.Sequential(self.scale, self.conv)

    def forward(self, x):
        return self.layer(x)


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=dim,
                        out_channels=dim // 4,
                        kernel_size=(1, 3, 3),
                        dilation=(1, rate, rate),
                        padding=(0, rate, rate),
                        padding_mode="replicate",
                    ),
                    nn.ReLU(True),
                ),
            )

        self.fuse = nn.Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(1, 3, 3),
            dilation=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )

        self.gate = nn.Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(1, 3, 3),
            dilation=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )

    def forward(self, x):
        out = [
            self.__getattr__(f"block{str(i).zfill(2)}")(x)
            for i in range(len(self.rates))
        ]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((1, 2, 3), keepdim=True)
    std = feat.std((1, 2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(
        self,
    ):
        super(Discriminator, self).__init__()
        inc = 1
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat
