import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision as tv

from models import Generator
from models import Discriminator
from torchvision import datasets
from torch.utils.data import DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=os.path.expanduser('~/.torch/datasets'))
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=64)
parser.add_argument('--out_channels', type=int, default=1)
cfg = parser.parse_args()
print(cfg)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

net_G = Generator(cfg.in_channels, cfg.out_channels)
net_G = net_G.to(device)
net_G.apply(weights_init)

net_D = Discriminator(cfg.out_channels)
net_D = net_D.to(device)
net_D.apply(weights_init)

dataset = datasets.MNIST(cfg.data, transform=tv.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
fixed_noises = torch.randn(cfg.batch_size, cfg.in_channels, 1, 1, device=device)

criterion = nn.MSELoss() # V(D, G) = log(D(x)) + log(1 - D(G(z)))
optimizer_G = torch.optim.Adam(net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

plt.ion()
for epoch in range(cfg.epochs):
    print('Epoch: {}/{}'.format(epoch + 1, cfg.epochs))

    for i, data in enumerate(dataloader):
        noises = torch.randn_like(fixed_noises, device=device)

        # -------------------
        # Train Discriminator
        # -------------------

        optimizer_D.zero_grad()

        # Update net_D: maximize D(x), minimize D(G(z)) -> maximize V(D, G)

        # train with real
        real = data[0].to(device)
        pred_real = net_D(real)
        loss_real = criterion(pred_real, torch.ones_like(pred_real, device=device))
        D_x = pred_real.mean().item()

        # train with fake
        fake = net_G(noises)
        pred_fake = net_D(fake.detach()) # not calc net_G's grad
        loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake, device=device))
        D_G_z1 = pred_fake.mean().item()

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # ---------------
        # Train Generator
        # ---------------

        optimizer_G.zero_grad()

        # Update net_G: maximize D(G(z)) -> minimize V(D, G)
        pred_fake = net_D(fake)
        loss_G = criterion(pred_fake, torch.ones_like(pred_fake, device=device))
        D_G_z2 = pred_fake.mean().item()

        loss_G.backward()
        optimizer_G.step()

        # ---------
        # Print Log
        # ---------

        print(
            '[{}/{}]'.format(epoch + 1, cfg.epochs) +
            '[{}/{}]'.format(i + 1, len(dataloader)) + ', ' +
            'loss_D: {:.4f}, loss_G: {:.4f}'.format(loss_D.item(), loss_G.item()) + ', ' +
            'D(x): {:.2f}, D(G(z)): {:.2f}/{:.2f}'.format(D_x, D_G_z1, D_G_z2)
        )

        if i % 100 == 99:
            fake = net_G(fixed_noises)
            fake = fake.detach().cpu()
            fake = tv.utils.make_grid(fake)
            fake = fake.numpy().transpose(1, 2, 0)
            plt.imshow(fake)
            plt.pause(0.1)

plt.ioff()
plt.show()
