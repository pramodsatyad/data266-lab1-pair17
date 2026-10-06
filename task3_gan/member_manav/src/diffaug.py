import torch
import torch.nn.functional as F


def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return (x - mean) * scale + mean


def rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - mean) * scale + mean


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    move_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    move_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + move_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + move_y + 1, 0, x.size(3) + 1)
    padded = F.pad(x, [1, 1, 1, 1])
    return padded.permute(0, 2, 3, 1).contiguous()[batch, grid_x, grid_y].permute(0, 3, 1, 2)


def rand_cutout(x, ratio=0.5):
    size_x, size_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - size_x % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - size_y % 2), size=[x.size(0), 1, 1], device=x.device)
    batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(size_x, device=x.device),
        torch.arange(size_y, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - size_x // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - size_y // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENTS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x, policy):
    for name in policy.split(","):
        for fn in AUGMENTS[name]:
            x = fn(x)
    return x.contiguous()
