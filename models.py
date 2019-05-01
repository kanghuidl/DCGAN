import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64):
        super(Generator, self).__init__()

        model = [
            # (in_channels, 1, 1)
            nn.ConvTranspose2d(in_channels, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # (ngf * 4, 4, 4)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # (ngf * 2, 8, 8)
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # (ngf, 16, 16)
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=4, stride=2, padding=3),
            nn.Sigmoid()
            # (out_channels, 28, 28)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=64):
        super(Discriminator, self).__init__()

        model = [
            # (in_channels, 28, 28)
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf, 16, 16)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 2, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 4, 4, 4)
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
            # (1, 1, 1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
