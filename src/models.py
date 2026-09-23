"""models.py — model zoo for the flood-segmentation benchmark (binary water/flood, 1 logit out).

  unet_r18 / unet_r34   U-Net with ResNet encoder (ResU-Net), smp
  deeplab_r34           DeepLabV3+ ResNet-34, smp
  segformer_b0 / b2     SegFormer (MiT encoder), smp
  attunet               Attention U-Net (Oktay et al. 2018), own implementation, no pretraining
Pretrained = ImageNet encoder weights (3-channel input VV, VH, VV-VH maps 1:1 onto RGB conv1).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class AttentionGate(nn.Module):
    """Additive attention gate: skip features x are re-weighted by a coefficient map computed from x and gate g."""
    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.wx = nn.Sequential(nn.Conv2d(f_x, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        a = self.psi(F.relu(self.wg(g) + self.wx(x), inplace=True))
        self.last_alpha = a.detach()                     # kept for attention-map figures
        return x * a


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, n_cls=1, base=32):
        super().__init__()
        f = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([_cbr(in_ch, f[0])] + [_cbr(f[i], f[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in reversed(range(4))])
        self.att = nn.ModuleList([AttentionGate(f[i], f[i], max(f[i] // 2, 8)) for i in reversed(range(4))])
        self.dec = nn.ModuleList([_cbr(2 * f[i], f[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(f[0], n_cls, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, att, dec in zip(self.up, self.att, self.dec):
            g = up(x); s = skips.pop()
            x = dec(torch.cat([att(g, s), g], 1))
        return self.head(x)


SPECS = {  # name: (builder, default lr, amp)
    "unet_r18":     ("smp.Unet", "resnet18", 1e-3, False),
    "unet_r34":     ("smp.Unet", "resnet34", 1e-3, False),
    "deeplab_r34":  ("smp.DeepLabV3Plus", "resnet34", 1e-3, False),
    "segformer_b0": ("smp.Segformer", "mit_b0", 3e-4, True),
    "segformer_b2": ("smp.Segformer", "mit_b2", 2e-4, True),
    "attunet":      ("own", None, 1e-3, False),
}


def build(name: str, in_ch=3, pretrained=True, classes=1):
    kind, enc, lr, amp = SPECS[name]
    if kind == "own":
        return AttentionUNet(in_ch, n_cls=classes), lr, amp
    import segmentation_models_pytorch as smp
    cls = {"smp.Unet": smp.Unet, "smp.DeepLabV3Plus": smp.DeepLabV3Plus, "smp.Segformer": smp.Segformer}[kind]
    m = cls(encoder_name=enc, encoder_weights="imagenet" if pretrained else None, in_channels=in_ch, classes=classes)
    return m, lr, amp
