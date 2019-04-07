import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision as tv


root = os.path.expanduser('~/.torch/datasets')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epochs = 20
batch_size = 64

nc, nz, ngf, ndf = 1, 64, 64, 64
fixed_noises = torch.randn(batch_size, nz, 1, 1, device=device)


transform = tv.transforms.Compose([tv.transforms.Resize(32), tv.transforms.ToTensor()])

dataset = tv.datasets.MNIST(root, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
    nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Sigmoid()
    # (nc, 32, 32)
)
gnet = gnet.to(device)
gnet.apply(weights_init)


dnet = nn.Sequential(
    # (nc, 32, 32)
    nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
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
    # (1, 1, 1)
)
dnet = dnet.to(device)
dnet.apply(weights_init)


criterion = nn.BCEWithLogitsLoss()
goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))


plt.ion()
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # train with real
        reals = data[0].to(device)
        batch_size = reals.size(0)

        labels = torch.ones(batch_size, device=device)
        outputs = dnet(reals).view(-1)
        dloss_real = criterion(outputs, labels)

        # train with fake
        noises = torch.randn(batch_size, nz, 1, 1, device=device)
        fakes = gnet(noises)
        labels = torch.zeros(batch_size, device=device)
        outputs = dnet(fakes.detach()).view(-1)
        dloss_fake = criterion(outputs, labels)

        dloss = dloss_real + dloss_fake
        dnet.zero_grad()
        dloss.backward()
        doptimizer.step()

        # (2) Update G network: maximize log(D(G(z)))
        labels = torch.ones(batch_size, device=device)
        outputs = dnet(fakes).view(-1)
        gloss = criterion(outputs, labels)
        gnet.zero_grad()
        gloss.backward()
        goptimizer.step()

        print(epoch, i, dloss.item(), gloss.item())

        if i % 100 == 0:
            images = gnet(fixed_noises)
            images = images.detach().cpu()
            images = tv.utils.make_grid(images)
            images = images.numpy().transpose(1, 2, 0)
            images[images < 0] = 0
            plt.imshow(images)
            plt.pause(0.1)

plt.ioff()
plt.show()
