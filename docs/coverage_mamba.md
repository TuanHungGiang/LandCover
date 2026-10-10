# Coverage-Preserving Quarter Mamba (C1)

## Contract

- Input/output: `B x C x H x W`, with even `H` and `W`.
- Context sequence: exactly `(H/2) x (W/2) = 25%` of the input spatial tokens.
- Every input pixel contributes to one context token; no top-k token is dropped.
- The original dense feature is retained by the outer residual in `GEE_Head`.
- Class logits provide detached soft prototype weights, never hard sequence membership.

## Data flow

```text
feature X --------------------------------------------------------------+
   |                                                                    |
   +-> X + zero-initialized depthwise 3x3                               |
       -> PixelUnshuffle(2)                                             |
       -> grouped 4-to-1 projection + channel-wise (max - mean) detail  |
       -> H/2 x W/2 context grid                                        |
       -> soft full-image class prototype conditioning                  |
       -> Fourier 2-D position                                          |
       -> serpentine forward/backward shared-weight SSDLite             |
       -> grouped 1-to-4 projection + PixelShuffle(2)                   |
       -> dense context ------------------------------------------------+-> X + context
```

## Weakness/compensation audit

| Mamba or compression weakness | Compensation in the complete model |
|---|---|
| 1-D scan breaks 2-D locality | Continuous serpentine order, two directions, Fourier 2-D position |
| Top-k omits confident errors and whole regions | Deterministic one-token-per-2x2 coverage |
| Argmax routing reinforces wrong classes | Full-image probability-weighted soft prototypes |
| 2x2 condensation attenuates small/high-frequency patterns | Pre-scan depthwise 3x3 plus max-minus-mean statistic |
| Context compression cannot be lossless at fixed channel width | Original dense feature remains on the residual path |
| Global context can blur boundaries | RepViT local features and stride-4 conv-only decoder path bypass Mamba |
| Long Mamba sequence is expensive | Exactly 25% context length at strides 8, 16 and 32 |
| Early auxiliary logits are unstable | Detached soft probabilities plus six-epoch bypass and six-epoch ramp |

## Fair ablation

The C1 config inherits the V1 backbone, loss, augmentation, optimizer, seed and 96-epoch schedule.
Only the scan block changes. Screen `full` inference first; run expensive TTA only if C1 beats V1's
`full=52.42` result. The final target remains V1's `tile512s256-ms=53.76`.
