import random

import torch
import torch.nn as nn


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.body(x)


class ResNetGenerator(nn.Module):
    def __init__(self, ngf=64, n_res_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        ch = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(True),
            ]
            ch *= 2
        layers += [ResBlock(ch) for _ in range(n_res_blocks)]
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(True),
            ]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class ImagePool:
    def __init__(self, size):
        self.size = size
        self.images = []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(image)
                out.append(image)
            elif random.random() < 0.5:
                i = random.randrange(self.size)
                out.append(self.images[i].clone())
                self.images[i] = image
            else:
                out.append(image)
        return torch.cat(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
