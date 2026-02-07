# Stage 2: Cross-Modality Alignment with Register Tokens

Owner: Taarush

## Goal

Align overlapping information across modalities (vision, tactile) while
preserving modality-specific (complementary) information, using FlexTok-inspired
register tokens with nested dropout.

## Architecture

```
Frozen Stage-1 Encoders
    [OpenCLIP ViT-L-14 (vision)] ──→ 768-d pooled features
    [TIMM ViT (tactile)]         ──→ 768-d pooled features
                                           │
                    ┌──────────────────────┘
                    ▼
        Register Token Module (per modality)
        ┌──────────────────────────────┐
        │ Input Projection (768→512)   │
        │ + Learnable Register Tokens  │
        │ + Causal Attention Masking   │
        │ + Nested Dropout (training)  │
        │ ──→ Shared Tokens [0:8]      │  ← aligned across modalities
        │ ──→ Private Tokens [8:32]    │  ← modality-specific info
        └──────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Shared Tokens              Private Tokens
  (mean pool + project)      (preserve modality info)
        │                       │
  Contrastive Loss          Preservation Loss
  (CLIP-style align)        (cosine similarity)
```

## Key Ideas

1. **Register Tokens** (from FlexTok, arXiv 2502.13967): Learnable embeddings that
   participate in self-attention with encoder features. Causal masking among registers
   enforces coarse-to-fine information ordering.

2. **Nested Dropout**: During training, randomly zero out trailing register tokens
   (power-of-two schedule: keep 1, 2, 4, 8, 16, or 32 tokens). This ensures each
   prefix is a valid representation at varying granularity.

3. **Token Splitting**: First `n_shared` registers capture cross-modal overlapping
   info (aligned via contrastive loss). Remaining registers preserve modality-specific
   complementary information (preserved via reconstruction/cosine loss).

4. **Flow Matching Alternative**: Optional rectified flow matching head that learns
   direct cross-modal transport (A→B, B→A) instead of / in addition to contrastive loss.

## Files

```
experiments/stage2/
├── README.md
├── __init__.py
├── train.py                          # Training script
├── configs/
│   └── default.yaml                  # Default hyperparameters
├── models/
│   ├── __init__.py
│   ├── register_tokens.py            # Register token module + nested dropout
│   └── cross_modal_alignment.py      # Full Stage 2 model + Stage2Wrapper
└── losses/
    ├── __init__.py
    ├── alignment_loss.py             # Contrastive + preservation losses
    └── flow_matching.py              # Rectified flow matching alternative
```

## Usage

```bash
# Single GPU training
python experiments/stage2/train.py \
    --stage1_checkpoint /path/to/stage1_best.pth \
    --datasets_dir /path/to/datasets \
    --output_dir ./output/stage2 \
    --epochs 100 \
    --batch_size 256 \
    --n_registers 32 \
    --n_shared 8

# With flow matching (combined loss)
python experiments/stage2/train.py \
    --stage1_checkpoint /path/to/stage1_best.pth \
    --datasets_dir /path/to/datasets \
    --loss_type combined \
    --flow_weight 0.5

# Multi-GPU
torchrun --nproc_per_node=4 experiments/stage2/train.py \
    --stage1_checkpoint /path/to/stage1_best.pth \
    --datasets_dir /path/to/datasets
```

## Hyperparameters to Explore

| Parameter | Default | Notes |
|-----------|---------|-------|
| n_registers | 32 | Total register tokens per modality |
| n_shared | 8 | Registers for cross-modal alignment |
| n_layers | 4 | Transformer depth in register module |
| hidden_dim | 512 | Internal dimension |
| nested_dropout_mode | power_of_two | vs. uniform sampling |
| loss_type | contrastive | vs. flow_matching, combined |
| preservation_weight | 0.5 | Weight for modality-specific preservation |

## References

- FlexTok: https://arxiv.org/abs/2502.13967
- Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588
- TiTok: https://arxiv.org/abs/2406.07550
- CrossFlow: https://arxiv.org/abs/2412.15213
- OmniFlow: https://arxiv.org/abs/2412.01169
