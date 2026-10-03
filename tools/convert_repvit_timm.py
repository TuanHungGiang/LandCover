"""Convert timm RepViT ImageNet weights (Hugging Face: timm/<model>.dist_450e_in1k, default repvit_m2_3)
into the key layout used by rsseg/models/backbones/repvit.py, so they can be used as
`pretrain/<model>_distill_450e.pth` (no Baidu account needed).

    python tools/convert_repvit_timm.py --dst pretrain/repvit_m2_3_distill_450e.pth
    python tools/convert_repvit_timm.py --model repvit_m1_1 --dst pretrain/repvit_m1_1_distill_450e.pth

Needs `safetensors` (pip install safetensors). The source file is downloaded from Hugging Face
if --src is not given.
"""
import argparse
import os
import re
import urllib.request

import torch

HF_URL = "https://huggingface.co/timm/{model}.dist_450e_in1k/resolve/main/model.safetensors"

BLOCK_RENAMES = [
    (r"^token_mixer\.conv\.", "token_mixer.0.conv."),
    (r"^token_mixer\.conv1\.", "token_mixer.0.conv1."),
    (r"^token_mixer\.bn\.", "token_mixer.0.bn."),
    (r"^se\.", "token_mixer.1."),
    (r"^channel_mixer\.conv1\.", "channel_mixer.m.0."),
    (r"^channel_mixer\.conv2\.", "channel_mixer.m.2."),
]
DOWNSAMPLE_RENAMES = [
    (r"^spatial_downsample\.", "token_mixer.0."),
    (r"^se\.", "token_mixer.1."),
    (r"^channel_downsample\.", "token_mixer.2."),
    (r"^ffn\.conv1\.", "channel_mixer.m.0."),
    (r"^ffn\.conv2\.", "channel_mixer.m.2."),
]


def _rename(rest, rules):
    for pat, rep in rules:
        if re.match(pat, rest):
            return re.sub(pat, rep, rest, count=1)
    raise KeyError(rest)


def convert(timm_sd):
    sd = {k: v for k, v in timm_sd.items() if not k.startswith("head")}

    # blocks per stage decide where each stage starts in the flat `features` list:
    # features.0 is the stem, then stage 0 blocks; every later stage adds
    # [pre_block, stride-2 block] before its own blocks.
    n_blocks = {}
    for k in sd:
        m = re.match(r"stages\.(\d+)\.blocks\.(\d+)\.", k)
        if m:
            n_blocks[int(m.group(1))] = max(n_blocks.get(int(m.group(1)), 0), int(m.group(2)) + 1)
    start = {0: 1}
    for s in range(1, len(n_blocks)):
        start[s] = start[s - 1] + n_blocks[s - 1] + 2

    out = {}
    for k, v in sd.items():
        m = re.match(r"stem\.conv(\d)\.(.*)", k)
        if m:
            out[f"features.0.{0 if m.group(1) == '1' else 2}.{m.group(2)}"] = v
            continue
        m = re.match(r"stages\.(\d+)\.blocks\.(\d+)\.(.*)", k)
        if m:
            s, j, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            out[f"features.{start[s] + j}.{_rename(rest, BLOCK_RENAMES)}"] = v
            continue
        m = re.match(r"stages\.(\d+)\.downsample\.pre_block\.(.*)", k)
        if m:
            s, rest = int(m.group(1)), m.group(2)
            out[f"features.{start[s] - 2}.{_rename(rest, BLOCK_RENAMES)}"] = v
            continue
        m = re.match(r"stages\.(\d+)\.downsample\.(.*)", k)
        if m:
            s, rest = int(m.group(1)), m.group(2)
            out[f"features.{start[s] - 1}.{_rename(rest, DOWNSAMPLE_RENAMES)}"] = v
            continue
        raise KeyError(k)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, help="timm model.safetensors (downloaded if omitted)")
    parser.add_argument("--model", default="repvit_m2_3", help="timm model name, e.g. repvit_m1_1 or repvit_m1_5")
    parser.add_argument("--dst", default=None, help="default: pretrain/<model>_distill_450e.pth")
    args = parser.parse_args()
    args.dst = args.dst or f"pretrain/{args.model}_distill_450e.pth"

    from safetensors.torch import load_file

    src = args.src
    if src is None:
        src = os.path.join(os.path.dirname(os.path.abspath(args.dst)) or ".", f"{args.model}_timm.safetensors")
        if not os.path.exists(src):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            url = HF_URL.format(model=args.model)
            print("downloading", url)
            urllib.request.urlretrieve(url, src)

    converted = convert(load_file(src))
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save({"model": converted}, args.dst)
    print(f"saved {len(converted)} tensors to {args.dst}")


if __name__ == "__main__":
    main()
