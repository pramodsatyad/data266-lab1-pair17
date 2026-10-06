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
    def __init__(self, ngf=64, n_res_blocks=9, upsample="transpose"):
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
            if upsample == "resize":
                layers += [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(ch, ch // 2, 3),
                    nn.InstanceNorm2d(ch // 2),
                    nn.ReLU(True),
                ]
            else:
                layers += [
                    nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(ch // 2),
                    nn.ReLU(True),
                ]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x, layers=None, encode_only=False):
        if layers is None:
            return self.model(x)
        feats = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in layers:
                feats.append(x)
        if encode_only:
            return feats
        return feats, x


class PatchSampleF(nn.Module):
    def __init__(self, feat_channels, nc=256):
        super().__init__()
        self.nc = nc
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(ch, nc), nn.ReLU(), nn.Linear(nc, nc)) for ch in feat_channels
        ])

    def forward(self, feats, num_patches=256, patch_ids=None):
        return_feats = []
        return_ids = []
        for i, feat in enumerate(feats):
            b, c, h, w = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            if patch_ids is not None:
                patch_id = patch_ids[i]
            else:
                patch_id = torch.randperm(h * w, device=feat.device)[:min(num_patches, h * w)]
            sample = feat[:, patch_id, :].reshape(-1, c)
            sample = self.mlps[i](sample)
            sample = sample / (sample.norm(dim=1, keepdim=True) + 1e-7)
            return_feats.append(sample)
            return_ids.append(patch_id)
        return return_feats, return_ids


class PatchNCELoss(nn.Module):
    def __init__(self, nce_t=0.07):
        super().__init__()
        self.nce_t = nce_t
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        n, dim = feat_q.shape
        l_pos = (feat_q * feat_k).sum(dim=1, keepdim=True)
        l_neg = feat_q @ feat_k.t()
        l_neg.fill_diagonal_(-10.0)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.nce_t
        target = torch.zeros(n, dtype=torch.long, device=feat_q.device)
        return self.cross_entropy(logits, target)


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf=64, spectral_norm=False):
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
        if spectral_norm:
            for i, layer in enumerate(self.model):
                if isinstance(layer, nn.Conv2d):
                    self.model[i] = nn.utils.spectral_norm(layer)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, ndf=64, spectral_norm=False, n_scales=2):
        super().__init__()
        self.nets = nn.ModuleList([PatchDiscriminator(ndf, spectral_norm) for _ in range(n_scales)])
        self.pool = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outs = []
        for net in self.nets:
            outs.append(net(x))
            x = self.pool(x)
        return outs


def build_discriminator(ndf=64, spectral_norm=False, d_scales=1):
    if d_scales == 2:
        return MultiScaleDiscriminator(ndf, spectral_norm)
    return PatchDiscriminator(ndf, spectral_norm)


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
