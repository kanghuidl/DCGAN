import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision as tv


epochs = 10
batch_size = 64

nc, nz, ngf, ndf = 1, 64, 64, 64

root = os.path.expanduser('~/.torch/datasets')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = tv.datasets.MNIST(root, transform=tv.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

fixed_noises = torch.randn(batch_size, nz, 1, 1).to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


gnet = nn.Sequential(
    # (nz, 1, 1)
    nn.ConvTranspose2d(nz, 4 * ngf, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(4 * ngf),
    nn.ReLU(inplace=True),
    # (4 * ngf, 4, 4)
    nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * ngf),
    nn.ReLU(inplace=True),
    # (2 * ngf, 8, 8)
    nn.ConvTranspose2d(2 * ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(ngf),
    nn.ReLU(inplace=True),
    # (ngf, 16, 16)
    nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=3, bias=False),
    nn.Sigmoid()
    # (nc, 28, 28)
)
gnet = gnet.to(device)
gnet.apply(weights_init)


dnet = nn.Sequential(
    # (nc, 28, 28)
    nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=3, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    # (ndf, 16, 16)
    nn.Conv2d(ndf, 2 * ndf, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * ndf),
    nn.LeakyReLU(0.2, inplace=True),
    # (2 * ndf, 8, 8)
    nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * ndf),
    nn.LeakyReLU(0.2, inplace=True),
    # (4 * ndf, 4, 4)
    nn.Conv2d(4 * ndf, 1, kernel_size=4, stride=1, padding=0, bias=False),
    nn.Sigmoid()
    # (1, 1, 1)
)
dnet = dnet.to(device)
dnet.apply(weights_init)


criterion = nn.BCELoss()
goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))

plt.ion()
for epoch in range(epochs):
    print('Epoch: {}/{}'.format(epoch + 1, epochs))

    for i, data in enumerate(dataloader):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # train with real
        real = data[0].to(device)
        batch_size = real.size(0)

        labels = torch.ones(batch_size).to(device)
        outputs = dnet(real).view(-1)
        dloss_real = criterion(outputs, labels)
        D_x = outputs.mean().item()

        # train with fake
        noises = torch.randn(batch_size, nz, 1, 1).to(device)
        fake = gnet(noises)
        labels = torch.zeros(batch_size).to(device)
        outputs = dnet(fake.detach()).view(-1) # not calc gnet's grad
        dloss_fake = criterion(outputs, labels)
        D_G_z1 = outputs.mean().item()

        dloss = dloss_real + dloss_fake
        dnet.zero_grad()
        dloss.backward()
        doptimizer.step()

        # (2) Update G network: maximize log(D(G(z)))
        labels = torch.ones(batch_size).to(device)
        outputs = dnet(fake).view(-1)
        gloss = criterion(outputs, labels)
        D_G_z2 = outputs.mean().item()

        gnet.zero_grad()
        gloss.backward()
        goptimizer.step()

        print(
            '[{}/{}]'.format(epoch + 1, epochs) +
            '[{}/{}]'.format(i + 1, len(dataloader)) + ', ' +
            'dloss: {:.4f}, gloss: {:.4f}'.format(dloss.item(), gloss.item()) + ', ' +
            'D(x): {:.2f}, D(G(z)): {:.2f}/{:.2f}'.format(D_x, D_G_z1, D_G_z2)
        )

        if i % 100 == 99:
            fake = gnet(fixed_noises)
            fake = fake.detach().cpu()
            fake = tv.utils.make_grid(fake)
            fake = fake.numpy().transpose(1, 2, 0)
            plt.imshow(fake)
            plt.pause(0.1)

plt.ioff()
plt.show()
