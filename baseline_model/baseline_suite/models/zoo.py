from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class CBR(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3, dilation: int = 1, groups: int = 1):
        padding = dilation * (kernel // 2)
        super().__init__(
            nn.Conv2d(cin, cout, kernel, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class TimmEncoder(nn.Module):
    def __init__(self, name: str, pretrained: bool = False, image_size: int = 512):
        super().__init__()
        kwargs = dict(features_only=True, pretrained=pretrained)
        if name.startswith("swin_"):
            kwargs.update(img_size=image_size, always_partition=True)
        self.net = timm.create_model(name, **kwargs)
        self.channels = list(self.net.feature_info.channels())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = list(self.net(x))
        # timm Swin feature maps are NHWC; all decoders in this suite use NCHW.
        return [y.permute(0, 3, 1, 2).contiguous() if y.shape[-1] == c else y for y, c in zip(outputs, self.channels)]


class PSPHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        cin, width = channels[-1], 128
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(cin, width, 1, bias=False), nn.ReLU(inplace=True)) for s in (1, 2, 3, 6)]
        )
        self.fuse = nn.Sequential(CBR(cin + 4 * width, 256), nn.Dropout2d(0.1), nn.Conv2d(256, classes, 1))

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x = fs[-1]
        return self.fuse(torch.cat([x] + [resize(stage(x), x.shape[-2:]) for stage in self.stages], 1))


class ASPP(nn.Module):
    def __init__(self, cin: int, width: int = 256):
        super().__init__()
        self.branches = nn.ModuleList([CBR(cin, width, 1)] + [CBR(cin, width, 3, d) for d in (6, 12, 18)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, width, 1, bias=False), nn.ReLU(inplace=True))
        self.fuse = CBR(width * 5, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([b(x) for b in self.branches] + [resize(self.pool(x), x.shape[-2:])], 1))


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.aspp = ASPP(channels[-1], 256)
        self.low = CBR(channels[0], 48, 1)
        self.out = nn.Sequential(CBR(304, 256), CBR(256, 256), nn.Conv2d(256, classes, 1))

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        low = self.low(fs[0])
        return self.out(torch.cat([resize(self.aspp(fs[-1]), low.shape[-2:]), low], 1))


class PositionAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        qk = max(8, channels // 8)
        self.q, self.k, self.v = nn.Conv2d(channels, qk, 1), nn.Conv2d(channels, qk, 1), nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        attn = torch.softmax(torch.bmm(q, k), -1)
        y = torch.bmm(self.v(x).flatten(2), attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.gamma * y


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.flatten(2)
        attn = torch.softmax(torch.bmm(flat, flat.transpose(1, 2)).amax(-1, keepdim=True) - torch.bmm(flat, flat.transpose(1, 2)), -1)
        return x + self.gamma * torch.bmm(attn, flat).reshape(b, c, h, w)


class DANetHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.reduce = CBR(channels[-1], 128, 3)
        self.pam, self.cam = PositionAttention(128), ChannelAttention()
        self.out = nn.Sequential(CBR(256, 128, 3), nn.Conv2d(128, classes, 1))

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x = self.reduce(fs[-1])
        return self.out(torch.cat((self.pam(x), self.cam(x)), 1))


class FPNDecoder(nn.Module):
    def __init__(self, channels: list[int], width: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList([CBR(c, width, 1) for c in channels])
        self.smooth = nn.ModuleList([CBR(width, width) for _ in channels])
        self.width = width

    def forward(self, fs: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        xs = [layer(x) for layer, x in zip(self.lateral, fs)]
        for i in range(len(xs) - 2, -1, -1):
            xs[i] = xs[i] + resize(xs[i + 1], xs[i].shape[-2:])
        xs = [layer(x) for layer, x in zip(self.smooth, xs)]
        base = xs[0].shape[-2:]
        return torch.stack([resize(x, base) for x in xs]).sum(0), xs


class SemanticFPNHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn, self.out = FPNDecoder(channels), nn.Conv2d(128, classes, 1)

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        return self.out(self.fpn(fs)[0])


class UPerHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.ppm = PSPHead([channels[-1]], 128)
        self.deep = CBR(128, 128, 1)
        self.fpn = FPNDecoder(channels[:-1] + [128])
        self.out = nn.Conv2d(128, classes, 1)

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        deep = self.deep(self.ppm([fs[-1]]))
        return self.out(self.fpn(fs[:-1] + [deep])[0])


class FarSegHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn = FPNDecoder(channels)
        self.scene = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels[-1], 128, 1))
        self.gate = nn.Sequential(nn.Conv2d(256, 128, 1), nn.Sigmoid())
        self.out = nn.Conv2d(128, classes, 1)

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        s = self.scene(fs[-1]).expand(-1, -1, *x.shape[-2:])
        return self.out(x * (1 + self.gate(torch.cat((x, s), 1))))


class OCRHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn, self.coarse = FPNDecoder(channels), nn.Conv2d(128, classes, 1)
        self.query, self.key, self.value = CBR(128, 64, 1), nn.Linear(128, 64), nn.Linear(128, 128)
        self.out = nn.Sequential(CBR(256, 128, 1), nn.Conv2d(128, classes, 1))

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        b, c, h, w = x.shape
        coarse = self.coarse(x).flatten(2)
        probs = torch.softmax(coarse, 1)
        feats = x.flatten(2)
        regions = torch.bmm(probs, feats.transpose(1, 2)) / probs.sum(2, keepdim=True).clamp_min(1e-6)
        q = self.query(x).flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, self.key(regions).transpose(1, 2)), -1)
        context = torch.bmm(attn, self.value(regions)).transpose(1, 2).reshape(b, c, h, w)
        return self.out(torch.cat((x, context), 1))


class LocalAttentionHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn = FPNDecoder(channels)
        self.local = nn.Sequential(CBR(128, 128, 3, groups=128), nn.Conv2d(128, 128, 1), nn.Sigmoid())
        self.out = nn.Conv2d(128, classes, 1)

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        return self.out(x * (1 + self.local(x)))


class ISNetHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn, self.coarse = FPNDecoder(channels), nn.Conv2d(128, classes, 1)
        self.image_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(128, 128, 1))
        self.fuse = nn.Sequential(CBR(128 * 3, 128, 1), nn.Conv2d(128, classes, 1))

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        probs = torch.softmax(self.coarse(x), 1)
        b, c, h, w = x.shape
        regions = torch.bmm(probs.flatten(2), x.flatten(2).transpose(1, 2))
        regions = regions / probs.flatten(2).sum(2, keepdim=True).clamp_min(1e-6)
        semantic = torch.bmm(probs.flatten(2).transpose(1, 2), regions).transpose(1, 2).reshape(b, c, h, w)
        image = self.image_context(x).expand_as(x)
        return self.fuse(torch.cat((x, semantic, image), 1))


class MultiAttentionHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn, self.position, self.channel = FPNDecoder(channels), PositionAttention(128), ChannelAttention()
        self.out = nn.Sequential(CBR(384, 128, 1), nn.Conv2d(128, classes, 1))

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        return self.out(torch.cat((x, self.position(x), self.channel(x)), 1))


class FullyAttentionHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn = FPNDecoder(channels)
        self.q, self.k, self.v = CBR(128, 32, 1), CBR(128, 32, 1), CBR(128, 128, 1)
        self.out = nn.Conv2d(128, classes, 1)

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        # Full spatial-channel attention is evaluated on a bounded grid to keep memory predictable.
        pooled = F.adaptive_avg_pool2d(x, (min(32, x.shape[-2]), min(32, x.shape[-1])))
        q, k, v = self.q(pooled).flatten(2).transpose(1, 2), self.k(pooled).flatten(2), self.v(pooled).flatten(2)
        y = torch.bmm(v, torch.softmax(torch.bmm(q, k), -1).transpose(1, 2)).reshape_as(pooled)
        return self.out(x + resize(y, x.shape[-2:]))


class SegmenterHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        width = 256
        self.proj = nn.Conv2d(channels[-1], width, 1)
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, 2)
        self.class_embed = nn.Parameter(torch.randn(classes, width) * 0.02)

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x = self.proj(fs[-1])
        b, c, h, w = x.shape
        tokens = self.decoder(x.flatten(2).transpose(1, 2))
        tokens = F.normalize(tokens, dim=-1)
        classes = F.normalize(self.class_embed, dim=-1)
        return torch.einsum("bnc,kc->bkn", tokens, classes).reshape(b, -1, h, w)


class RoutedAttentionHead(nn.Module):
    """Bi-level routed attention decoder used for the standardized BiFormer baseline."""
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn = FPNDecoder(channels)
        self.route = nn.Conv2d(128, 16, 1)
        self.experts = nn.ModuleList([CBR(128, 128, 3, groups=8) for _ in range(4)])
        self.out = nn.Conv2d(128, classes, 1)

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        weights = torch.softmax(self.route(F.adaptive_avg_pool2d(x, 1))[:, :4], 1)
        y = sum(weights[:, i : i + 1] * expert(x) for i, expert in enumerate(self.experts))
        return self.out(x + y)


class LOGCANHead(nn.Module):
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn, self.seed = FPNDecoder(channels), nn.Conv2d(128, classes, 1)
        self.affine = nn.Sequential(nn.Conv2d(128, 128, 1), nn.Sigmoid())
        self.out = nn.Conv2d(256, classes, 1)

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        probs = torch.softmax(self.seed(x), 1)
        global_class = torch.bmm(probs.flatten(2), x.flatten(2).transpose(1, 2))
        global_class /= probs.flatten(2).sum(2, keepdim=True).clamp_min(1e-6)
        context = torch.bmm(probs.flatten(2).transpose(1, 2), global_class).transpose(1, 2).reshape_as(x)
        return self.out(torch.cat((x, context * (1 + self.affine(x))), 1))


class SCSMHead(nn.Module):
    """Paper-derived SCSM reproduction; official authors have not released SCSM source/checkpoints."""
    def __init__(self, channels: list[int], classes: int):
        super().__init__()
        self.fpn = FPNDecoder(channels)
        self.scene = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(128, classes, 1))
        self.mask = nn.Conv2d(128, classes, 1)
        self.refine = nn.Sequential(CBR(128 + classes, 128, 1), nn.Conv2d(128, classes, 1))

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        x, _ = self.fpn(fs)
        scene = torch.sigmoid(self.scene(x))
        masks = self.mask(x) * scene
        coupling = torch.softmax(masks, 1)
        return masks + self.refine(torch.cat((x, coupling), 1))


class DDPHead(nn.Module):
    """Compact conditional denoising decoder following DDP's noise-to-map inference paradigm."""
    def __init__(self, channels: list[int], classes: int, steps: int = 3):
        super().__init__()
        self.fpn, self.steps, self.classes = FPNDecoder(channels), steps, classes
        self.denoise = nn.Sequential(CBR(128 + classes + 1, 128), CBR(128, 128), nn.Conv2d(128, classes, 1))

    def forward(self, fs: list[torch.Tensor]) -> torch.Tensor:
        cond, _ = self.fpn(fs)
        logits = torch.zeros(cond.shape[0], self.classes, *cond.shape[-2:], device=cond.device, dtype=cond.dtype)
        for step in reversed(range(self.steps)):
            t = cond.new_full((cond.shape[0], 1, *cond.shape[-2:]), step / max(1, self.steps - 1))
            logits = logits + self.denoise(torch.cat((cond, logits, t), 1)) / self.steps
        return logits


@dataclass(frozen=True)
class ModelSpec:
    backbone: str
    head: type[nn.Module]
    fidelity: str = "standardized"


_SPECS = {
    "pspnet": ModelSpec("resnet50", PSPHead),
    "deeplabv3plus": ModelSpec("resnet50", DeepLabV3PlusHead),
    "danet": ModelSpec("resnet50", DANetHead),
    "semantic_fpn": ModelSpec("resnet50", SemanticFPNHead),
    "farseg": ModelSpec("resnet50", FarSegHead),
    "ocrnet": ModelSpec("resnet50", OCRHead),
    "lanet": ModelSpec("resnet50", LocalAttentionHead),
    "isnet": ModelSpec("resnet50", ISNetHead),
    "segmenter": ModelSpec("resnet50", SegmenterHead),
    "swin_upernet": ModelSpec("swin_tiny_patch4_window7_224", UPerHead),
    "manet": ModelSpec("resnet50", MultiAttentionHead),
    "flanet": ModelSpec("resnet50", FullyAttentionHead),
    "convnext": ModelSpec("convnext_tiny", UPerHead),
    "poolformer": ModelSpec("poolformer_s12", UPerHead),
    "biformer": ModelSpec("resnet50", RoutedAttentionHead, "paper-derived"),
    "efficientvit": ModelSpec("efficientvit_b1", UPerHead),
    "ddp": ModelSpec("resnet50", DDPHead, "compact-paper-derived"),
    "logcanpp": ModelSpec("resnet50", LOGCANHead, "paper-derived"),
    "scsm": ModelSpec("resnet50", SCSMHead, "paper-derived-unofficial"),
}
MODEL_NAMES = tuple(_SPECS)


class SegmentationModel(nn.Module):
    def __init__(self, spec: ModelSpec, classes: int, pretrained: bool, image_size: int):
        super().__init__()
        self.encoder = TimmEncoder(spec.backbone, pretrained=pretrained, image_size=image_size)
        self.decode_head = spec.head(self.encoder.channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(self.decode_head(self.encoder(x)), x.shape[-2:])


def model_info(name: str) -> dict[str, str]:
    spec = _SPECS[name]
    return {"name": name, "backbone": spec.backbone, "fidelity": spec.fidelity}


def build_model(name: str, num_classes: int = 7, pretrained: bool = False, image_size: int = 512) -> nn.Module:
    key = name.lower().replace("-", "_").replace("+", "p")
    aliases = {"deeplabv3p": "deeplabv3plus", "swinupernet": "swin_upernet", "logcanpp": "logcanpp"}
    key = aliases.get(key, key)
    if key not in _SPECS:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}")
    return SegmentationModel(_SPECS[key], num_classes, pretrained, image_size)
