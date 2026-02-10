# CURSOR IMPLEMENTATION - FluxTalk Register Tokens

## Experiment Metadata

**Created by**: Cursor (Composer)  
**Model**: Composer (Cursor's AI coding assistant)  
**Date**: February 7, 2026  
**Experiment ID**: `cursor_flux_talk_register_tokens`  
**Implementation Version**: 1.0

## Overview

This is **Cursor's implementation** of Stage 2 cross-modality alignment using FluxTalk-style register tokens with nested dropout. This implementation is one of several experimental approaches being tested for Stage 2.

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

### Import the Module

```python
from experiments.stage2.cursor.register_tokens import RegisterTokenModule
from experiments.stage2.cursor.loss import RegisterTokenLoss
```

### Training

```bash
cd experiments/stage2/cursor

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 train.py \
    --stage1_checkpoint /path/to/stage1/checkpoint.pth \
    --batch_size 64 \
    --epochs 100 \
    --num_register_tokens 4 \
    --num_overlapping_tokens 3 \
    --nested_dropout_rate 0.5 \
    --complementary_reg_weight 0.01 \
    --output_dir ./output_dir/cursor_experiment \
    --log_name cursor_flux_talk \
    [other args...]
```

## Files

- `register_tokens.py`: `RegisterTokenModule` class - Model implementation
- `loss.py`: `RegisterTokenLoss` class - Loss function
- `engine.py`: Training/evaluation loops
- `train.py`: Main training script
- `__init__.py`: Package initialization
- `README.md`: This file
- `METADATA.txt`: Quick metadata reference

## Identifying This Implementation

All files include clear Cursor identification:
- File headers with "CURSOR IMPLEMENTATION" marker
- Experiment ID: `cursor_flux_talk_register_tokens`
- Logs and outputs labeled with `[CURSOR]` prefix
- All results include `'implementation': 'cursor'` in logs

## Comparison with Other Implementations

When comparing results:
- Look for `[CURSOR]` prefix in logs
- Check for `'implementation': 'cursor'` in log files
- Verify experiment ID matches `cursor_flux_talk_register_tokens`
- Compare hyperparameters (especially register token configuration)

## Known Issues / Limitations

- Register tokens are cloned for each modality (memory overhead)
- Complementary token regularization mask logic may need tuning
- Default hyperparameters may need adjustment based on dataset

## Next Steps

1. Test imports: `python experiments/stage2/test_modules.py`
2. Run training with your Stage 1 checkpoint
3. Compare results with other implementations
4. Tune hyperparameters as needed
