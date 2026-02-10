# FluxTalk Register Tokens - Stage 2 Variant

## Experiment Metadata

**Created by**: Cursor (Composer)  
**Model**: Composer (Cursor's AI coding assistant)  
**Date**: February 7, 2026  
**Experiment ID**: `flux_talk_register_tokens`  
**Implementation Version**: 1.0

## Approach

- **Method**: FluxTalk register tokens with nested dropout
- **Register Tokens**: 4 tokens total (3 overlapping + 1 complementary)
- **Alignment**: Contrastive learning on overlapping tokens
- **Training**: Nested dropout for hierarchical information capture

## Key Features

1. Learnable register tokens that attend to modality features via cross-attention
2. Nested dropout schedule (increasing probability for later tokens)
3. Per-modality token processing for cross-modal alignment
4. Complementary token preservation with regularization

## Design Decisions

- **Per-modality processing**: Register tokens are processed separately for each modality to enable cross-modal alignment
- **Nested dropout**: Linear schedule from 0% to `nested_dropout_rate` across tokens
- **Token allocation**: 3 overlapping tokens (majority) + 1 complementary token
- **Loss aggregation**: Average pooling of overlapping tokens before contrastive loss

## Implementation Notes

- Stage 1 encoders are frozen (no gradients)
- Only register tokens and cross-attention layers are trainable
- Nested dropout applied only during training (disabled during eval)
- Complementary tokens use L2 + diversity regularization

## Usage

```bash
cd experiments/stage2/variants/flux_talk_register_tokens

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 main_train_stage2.py \
    --stage1_checkpoint /path/to/stage1/checkpoint.pth \
    --batch_size 64 \
    --epochs 100 \
    --num_register_tokens 4 \
    --num_overlapping_tokens 3 \
    --nested_dropout_rate 0.5 \
    --complementary_reg_weight 0.01 \
    --output_dir ./output_dir/flux_talk_variant \
    [other args...]
```

## Files

- `tvl_stage2.py`: Model implementation
- `loss_stage2.py`: Loss function
- `engine_stage2.py`: Training/evaluation loops
- `main_train_stage2.py`: Main training script

## Known Issues / Limitations

- Register tokens are cloned for each modality (memory overhead)
- Complementary token regularization mask logic may need tuning
- Default hyperparameters may need adjustment based on dataset
