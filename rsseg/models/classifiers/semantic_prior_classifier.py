import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsseg.models.segheads.gee_head import entropy_gate


class SemanticPrior_Classifier(nn.Module):
    """Base_Classifier plus a frozen CLIP/RemoteCLIP class-prototype prior, mixed in only for
    pixels the base classifier is not confident about (same entropy gate as GEE_Head's exploit/explore
    routing). Targets classes whose name is ambiguous in isolation (e.g. LoveDA's "barren" vs "bareland",
    "rangeland" -- see tools/build_clip_prototypes.py) by giving the model a class prior tied to several
    synonym prompts instead of a single learned one-hot center.

    final_logits = base_logits + beta * (1 - g) * cosine_similarity(proj(feat), prototypes) / temperature

    `prototypes_path` must exist (build it once with tools/build_clip_prototypes.py); `beta` starts small
    and is learned, so the prior can contribute little if it turns out not to help.
    """

    def __init__(self, transform_channel, num_class, prototypes_path, beta=0.1, temperature=0.07):
        super().__init__()
        self.classifier = nn.Conv2d(transform_channel, num_class, kernel_size=1, stride=1)

        if not os.path.isfile(prototypes_path):
            raise FileNotFoundError(
                f"{prototypes_path} not found. Build it once with:\n"
                f"    python tools/build_clip_prototypes.py --out {prototypes_path}")
        ckpt = torch.load(prototypes_path, map_location='cpu')
        prototypes = ckpt['prototypes'] if isinstance(ckpt, dict) else ckpt
        assert prototypes.shape[0] == num_class, \
            f"{prototypes_path} has {prototypes.shape[0]} class prototypes, expected {num_class}"
        self.register_buffer('prototypes', F.normalize(prototypes.float(), dim=-1))  # (num_class, D), frozen

        self.proj = nn.Conv2d(transform_channel, prototypes.shape[1], kernel_size=1)
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.temperature = temperature

    def forward(self, out):
        feat = out[0]
        base_logits = self.classifier(feat)

        proj = F.normalize(self.proj(feat), dim=1)                              # (B, D, H, W)
        sim = torch.einsum('bdhw,kd->bkhw', proj, self.prototypes) / self.temperature  # (B, num_class, H, W)

        g = entropy_gate(base_logits).to(feat.dtype)
        pred = base_logits + self.beta * (1. - g) * sim
        return [pred] + out[1:]
