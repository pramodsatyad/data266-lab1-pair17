"""From-scratch CycleGAN networks. No pretrained generation models.

Architecture reference: Zhu et al., ICCV 2017 (https://arxiv.org/abs/1703.10593).
A = Monet; B = Photo. D_A judges Monet and D_B judges Photo.
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    def __init__(self, channels=64, blocks=9, upsample="transpose"):
        super().__init__()
        if upsample not in {"transpose", "resize"}:
            raise ValueError("upsample must be transpose or resize")
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, channels, 7),
                  nn.InstanceNorm2d(channels), nn.ReLU(True)]
        c = channels
        for _ in range(2):
            layers += [nn.Conv2d(c, c * 2, 3, 2, 1),
                       nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
            c *= 2
        layers += [ResidualBlock(c) for _ in range(blocks)]
        for _ in range(2):
            if upsample == "transpose":
                layers += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1)]
            else:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                           nn.ReflectionPad2d(1), nn.Conv2d(c, c // 2, 3)]
            c //= 2
            layers += [nn.InstanceNorm2d(c), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh()]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class PatchDiscriminator(nn.Module):
    """70x70 receptive field; logits (no sigmoid) for least-squares GAN."""
    def __init__(self, channels=64):
        super().__init__()
        layers = [nn.Conv2d(3, channels, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        for stride in [2, 2, 1]:
            layers += [nn.Conv2d(channels, channels * 2, 4, stride, 1),
                       nn.InstanceNorm2d(channels * 2), nn.LeakyReLU(0.2, True)]
            channels *= 2
        layers += [nn.Conv2d(channels, 1, 4, 1, 1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_models(config, device):
    spec = config["model"]
    models = {"G_A2B": Generator(**spec), "G_B2A": Generator(**spec),
              "D_A": PatchDiscriminator(spec["channels"]),
              "D_B": PatchDiscriminator(spec["channels"])}
    for model in models.values():
        model.apply(initialize)
        model.to(device)
    return models
