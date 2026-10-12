# C2: exact-budget adaptive spectral Landscape Mamba

C2 is an ablation-safe successor to the uniform C1 coverage block. It leaves the RepViT backbone,
losses, augmentation and optimizer unchanged.

## Conservation rules

1. Full coverage: adaptive regions form a disjoint partition of the feature map.
2. Exact cardinality: one 4-to-1 homogeneous merge is paired with one 1-to-4 complex split, hence
   `N - 3K + 3K = N = H*W/4` context tokens.
3. Pixel mass: region masses are 1, 4 or 16 pixels and always sum to `H*W`.

Complexity combines local feature variance, boundary magnitude, prediction entropy and scene-relative
rarity. Semantic terms follow the existing warm-up/ramp so unreliable early predictions do not dominate
routing.

## Spectral traversal

A sparse four-neighbour graph is weighted by feature and probability differences. Deflated power
iterations on the normalized adjacency approximate its first non-trivial eigenfunction in linear space
and time. Each adaptive region receives the pixel-mass mean of this field. Sorting that scalar yields a
content-aware forward order; the same SSM weights process forward and reverse sequences.

This is not claimed as the first Fiedler/spectral Mamba scan. The research contribution to test is the
combination of exact compute conservation, full coverage and area-aware ordering of mixed-resolution
land-cover regions.

## Configuration and checks

```bash
python tools/check_adaptive_spectral_scan.py
python tools/benchmark_coverage_blocks.py --size 64 --batch 2 --amp
python train.py -c configs/loveda/gee_repvit_landcover_adaptive_spectral.py
```

Primary ablations should vary `adapt_fraction` while holding `ratio=0.25`; compare C1 uniform tokens,
random paired redistribution, complexity-only paired redistribution and the complete spectral version.

## Where metaheuristics fit

Use an offline multi-objective search only after the mechanism is validated. A chromosome can contain
token budget, `adapt_fraction`, routing weights, spectral iterations and temperature. Optimize validation
mIoU jointly with measured latency and peak memory, and report the full Pareto front. The final test set
must not participate in this search. Metaheuristic tuning alone is not architectural novelty; a stronger
claim is a budget controller that transfers unchanged across datasets or hardware.
