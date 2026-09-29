# Adds SemanticPrior_Classifier (CLIP/RemoteCLIP class-prototype prior, mixed in only for pixels the
# base classifier is unsure about) on top of gee_repvit.py. Needs pretrain/clip_prototypes.pt:
#     pip install open_clip_torch huggingface_hub
#     python tools/build_clip_prototypes.py --out pretrain/clip_prototypes.pt
_base_ = './gee_repvit.py'

exp_name = "work_dirs/gee_repvit_semantic_loveda"
model_config = dict(
    classifier = dict(
        type = 'SemanticPrior_Classifier',
        transform_channel = 96,
        num_class = 7,
        prototypes_path = 'pretrain/clip_prototypes.pt',
        beta = 0.1,
        temperature = 0.07,
    ),
)
