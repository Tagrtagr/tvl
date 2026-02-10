# Stage 2: Cross-Modality Alignment with Register Tokens

## Overview

Stage 2 implements cross-modality alignment using FluxTalk-style register tokens. This stage focuses on aligning overlapping information across modalities (vision, tactile, text) while preserving complementary modality-specific information.

## Experiment Implementations

Multiple experimental approaches are being tested. Each implementation is in its own subdirectory for easy comparison.

### Available Implementations

1. **Cursor Implementation** (`cursor/`)
   - **Created by**: Cursor (Composer)
   - **Experiment ID**: `cursor_flux_talk_register_tokens`
   - **Classes**: `RegisterTokenModule`, `RegisterTokenLoss`
   - **Import**: `from experiments.stage2.cursor.register_tokens import RegisterTokenModule`
   - See [cursor/README.md](cursor/README.md) for details

2. **Variants** (`variants/`)
   - Additional experimental approaches
   - See [variants/README.md](variants/README.md) for details

### Testing Imports

Test that implementations can be imported:
```bash
python experiments/stage2/test_modules.py
```

### Adding New Implementations

When adding a new implementation (e.g., from Claude, OpenAI, etc.):
1. Create a new subdirectory (e.g., `claude/`, `openai/`)
2. Include clear metadata headers in all files
3. Use consistent naming conventions
4. Update this README
5. Add tests to `test_modules.py`

## Goal

- Align overlapping information across modalities using register tokens
- Preserve complementary information using dedicated register tokens
- Keep Stage 1 encoders frozen (no retraining)
- Use contrastive learning for alignment
- Apply nested dropout for robust token learning

## Architecture

### Components

1. **Frozen Stage 1 Encoders**: Pre-trained TVL encoders that extract features from each modality
2. **Register Tokens**: Learnable embedding vectors that capture information at different granularities
   - 4 total tokens (configurable)
   - 3 tokens for overlapping cross-modal information
   - 1 token for complementary modality-specific information
3. **Cross-Attention Layers**: Transformer blocks that allow register tokens to attend to modality features
4. **Nested Dropout**: Training technique that drops tokens with increasing probability to force hierarchical information capture

### Architecture Flow

```
Input (vision, tactile, text)
    ↓
Stage 1 Encoders (FROZEN)
    ↓
Modality Features (vision_feat, tactile_feat, text_feat)
    ↓
Register Tokens + Cross-Attention
    ↓
Per-Modality Token Representations
    ↓
Overlapping Tokens (aligned via contrastive loss)
Complementary Tokens (preserved per modality)
```

## Implementation Details

### Files

- `tvl_stage2.py`: Main Stage 2 model class (`TVLStage2`)
- `loss_stage2.py`: Loss function for Stage 2 (`Stage2Loss`)
- `engine_stage2.py`: Training and evaluation loops
- `main_train_stage2.py`: Main training script

### Key Features

1. **Register Tokens**
   - Learnable parameters initialized with small random values
   - Dimension matches Stage 1 encoder output (default: 768)
   - Processed separately per modality to get modality-specific representations

2. **Cross-Attention**
   - Query: Register tokens
   - Key/Value: Modality features from Stage 1
   - Multiple transformer layers (default: 2)
   - Self-attention enabled after first layer

3. **Nested Dropout**
   - Dropout probability increases for later tokens
   - Token 0: 0% dropout (always kept)
   - Token 1: ~17% dropout
   - Token 2: ~33% dropout
   - Token 3: ~50% dropout (configurable via `nested_dropout_rate`)
   - Forces earlier tokens to capture coarse information

4. **Loss Function**
   - Contrastive loss on overlapping tokens across modalities
   - Optional regularization on complementary tokens to maintain diversity
   - Computes accuracy metrics for each modality pair

## Usage

### Using Cursor Implementation

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

### Using Base Implementation

```bash
cd experiments/stage2

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 main_train_stage2.py \
    --stage1_checkpoint /path/to/stage1/checkpoint.pth \
    --batch_size 64 \
    --epochs 100 \
    --warmup_epochs 10 \
    --weight_decay 0.05 \
    --datasets ssvtp hct \
    --active_modality_names vision tactile text \
    --find_unused_parameters \
    --multi_epochs_dataloader \
    --log_name stage2_experiment \
    --shuffle_text \
    --no_text_prompt \
    --replace_synonyms \
    --num_workers 20 \
    --use_not_contact \
    --tactile_model vit_tiny_patch16_224 \
    --blr 1e-3 \
    --datasets_dir /path/to/datasets \
    --num_register_tokens 4 \
    --num_overlapping_tokens 3 \
    --register_token_dim 768 \
    --num_cross_attn_layers 2 \
    --nested_dropout_rate 0.5 \
    --complementary_reg_weight 0.01 \
    --output_dir ./output_dir
```

### Key Arguments

**Stage 1 Checkpoint:**
- `--stage1_checkpoint`: Path to Stage 1 model checkpoint (required)

**Register Token Configuration:**
- `--num_register_tokens`: Total number of register tokens (default: 4)
- `--num_overlapping_tokens`: Tokens for overlapping info (default: 3)
- `--register_token_dim`: Dimension of register tokens (default: 768)
- `--num_cross_attn_layers`: Number of cross-attention layers (default: 2)
- `--nested_dropout_rate`: Base dropout rate for nested dropout (default: 0.5)

**Loss Configuration:**
- `--complementary_reg_weight`: Weight for complementary token regularization (default: 0.01)
- `--disable_vision_text_loss`: Disable vision-text alignment loss
- `--disable_tactile_text_loss`: Disable tactile-text alignment loss

**Model Architecture:**
- `--num_heads`: Number of attention heads (default: 8)
- `--mlp_ratio`: MLP expansion ratio (default: 4.0)
- `--drop_rate`: Dropout rate (default: 0.0)
- `--attn_drop_rate`: Attention dropout rate (default: 0.0)
- `--drop_path_rate`: Drop path rate (default: 0.0)

**Stage 1 Model Parameters (must match Stage 1 training):**
- `--tactile_model`: Tactile encoder model (default: vit_tiny_patch16_224)
- `--common_latent_dim`: Common latent dimension if used in Stage 1 (default: None)

### Evaluation

The training script automatically evaluates on the validation set after each epoch. Metrics logged include:
- Overlapping token alignment loss per modality pair
- Accuracy (Acc@1, Acc@5) for each modality pair
- Complementary token regularization loss
- Total loss

## Technical Details

### Register Token Initialization

Register tokens are initialized as:
```python
nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
```

### Nested Dropout Schedule

Dropout probabilities are computed as:
```python
dropout_probs = torch.linspace(0.0, nested_dropout_rate, num_tokens)
```

This creates a linear schedule where later tokens have higher dropout probability.

### Loss Computation

1. **Overlapping Token Loss**: Average pooling of overlapping tokens → contrastive loss across modalities
2. **Complementary Token Regularization**: L2 norm + diversity penalty (optional)
3. **Total Loss**: `overlapping_loss + complementary_reg_weight * complementary_reg`

### Model Output

The model returns a dictionary with:
- `overlapping_tokens`: Dict mapping modality to overlapping tokens `(batch_size, num_overlapping_tokens, token_dim)`
- `complementary_tokens`: Dict mapping modality to complementary tokens `(batch_size, num_complementary_tokens, token_dim)`
- `register_tokens`: All register tokens `(batch_size, num_register_tokens, token_dim)`
- `stage1_features`: Original Stage 1 features
- `nested_dropout_mask`: Boolean mask indicating which tokens were kept (training only)

## Scope

- ✅ Frozen Stage-1 encoders
- ✅ Learnable register tokens
- ✅ Contrastive alignment on overlapping tokens
- ✅ Complementary token preservation
- ✅ Nested dropout for robust training
- ❌ No downstream task (Stage 3)

## Owner

Taarush

## References

- FluxTalk: Register Tokens for Image Generation
- Nested Dropout for Hierarchical Representation Learning
- Cross-Modality Contrastive Learning
