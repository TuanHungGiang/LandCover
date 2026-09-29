"""Build one CLIP text-prototype vector per LoveDA class from several synonym prompts, so an
ambiguously-named class (e.g. "barren") is anchored to more than one phrasing instead of a single
learned one-hot center. Saves a small (num_class, D) tensor consumed by
rsseg/models/classifiers/semantic_prior_classifier.py -- no CLIP model is needed at train/inference
time, only this file.

Needs: pip install open_clip_torch huggingface_hub

    python tools/build_clip_prototypes.py --out pretrain/clip_prototypes.pt
"""
import argparse
import os

import torch

# Order must match configs/_base_/loveda_config.py's class_name list.
# Several phrasings per class where the literature flags single-word ambiguity
# (e.g. "barren"/"bareland"/"rangeland" sit far apart in CLIP's text embedding space).
LOVEDA_PROMPTS = {
    'building': ['a building', 'a house', 'a rooftop', 'an urban building'],
    'road': ['a road', 'a street', 'a paved road', 'a highway'],
    'water': ['water', 'a river', 'a lake', 'a body of water'],
    'barren': ['barren land', 'bare land', 'bare soil', 'exposed ground', 'rangeland', 'an empty lot'],
    'forest': ['a forest', 'trees', 'woodland', 'dense vegetation'],
    'agricultural': ['farmland', 'a cropland', 'an agricultural field', 'a plowed field'],
    'background': ['background', 'an unlabeled area', 'unclassified terrain'],
}
PROMPT_TEMPLATES = ['{}', 'a satellite image of {}', 'an aerial photo of {}', 'a remote sensing image showing {}']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='chendelong/RemoteCLIP', help='Hugging Face repo id')
    parser.add_argument('--model', default='ViT-B-32', help='open_clip architecture name')
    parser.add_argument('--filename', default=None, help='checkpoint filename in the HF repo (default: RemoteCLIP-<model>.pt)')
    parser.add_argument('--class-names', nargs='+', default=list(LOVEDA_PROMPTS.keys()),
                         help='must match configs/_base_/loveda_config.py class_name order')
    parser.add_argument('--out', default='pretrain/clip_prototypes.pt')
    args = parser.parse_args()

    import open_clip
    from huggingface_hub import hf_hub_download

    filename = args.filename or f'RemoteCLIP-{args.model}.pt'
    ckpt_path = hf_hub_download(args.repo, filename)

    model, _, _ = open_clip.create_model_and_transforms(args.model)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    prototypes = []
    with torch.no_grad():
        for cls in args.class_names:
            phrases = [t.format(p) for p in LOVEDA_PROMPTS[cls] for t in PROMPT_TEMPLATES]
            tokens = tokenizer(phrases)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            prototypes.append(feats.mean(dim=0))                # average synonym/template embeddings
            print(f"{cls:<14} {len(phrases)} prompts")

    prototypes = torch.stack(prototypes)                        # (num_class, D), not yet re-normalized
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'prototypes': prototypes, 'classes': args.class_names, 'repo': args.repo, 'model': args.model}, args.out)
    print(f"saved {tuple(prototypes.shape)} prototypes to {args.out}")


if __name__ == '__main__':
    main()
