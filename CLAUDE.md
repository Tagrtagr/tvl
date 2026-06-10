# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Touch-Vision-Language (TVL)** codebase for multimodal alignment of tactile, vision, and language modalities. The work has two phases:

1. **Stage 1 (TVL encoder)**: Train a tactile ViT encoder (ViT-Tiny/Small/Base) to align with CLIP's latent space using contrastive learning across vision, tactile, and text modalities.
2. **Stage 2 (experiments/)**: Train cross-modal register tokens on top of frozen Stage-1 encoders. The active experimental approach (`experiments/stage2/claude_flextok/`) uses FlexTok-inspired register tokens with nested dropout.

## Environment Setup

```bash
conda create -n tvl python=3.10 -y
conda activate tvl
conda install pytorch==2.1.2 cudatoolkit==11.8.0 -c pytorch -y
pip install packaging
pip install -r requirements.txt
pip install -e .
```

## Key Commands

### Smoke Test (no GPU needed)
```bash
python experiments/stage2/claude_flextok/test_modules.py
```

### Stage 2 Training (Hydra-based entrypoint)
All training is launched from the repo root. The entrypoint uses Hydra with config groups in `experiments/stage2/claude_flextok/configs/`.

```bash
# Stage 2a: alignment (contrastive + flow-matching probe, default config)
python experiments/stage2/claude_flextok/train.py \
    stage=alignment \
    stage1_checkpoint=/path/to/tvl_enc.pth \
    datasets_dir=/path/to/data \
    output_dir=./output/stage2a \
    log_name=run_name

# Stage 2b: reconstruction decoder (frozen alignment, AR decoder)
python experiments/stage2/claude_flextok/train.py \
    stage=reconstruction \
    model=reconstruction \
    losses=reconstruction \
    alignment_checkpoint=./output/stage2a/run_name/checkpoint_best.pth \
    stage1_checkpoint=/path/to/tvl_enc.pth \
    datasets_dir=/path/to/data \
    output_dir=./output/stage2b

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 experiments/stage2/claude_flextok/train.py \
    stage=alignment stage1_checkpoint=... datasets_dir=...

# Resume training
python experiments/stage2/claude_flextok/train.py resume=auto ...
```

Hydra config overrides use `key=value` syntax (not `--key value`), except `--resume` which is normalized to Hydra style automatically.

### Stage 1 Tactile Encoder Training
```bash
cd tvl_enc
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 main_pretrain.py \
    --batch_size 256 --epochs 200 --warmup_epochs 10 --weight_decay 0.05 \
    --datasets ssvtp hct --active_modality_names vision tactile text \
    --find_unused_parameters --multi_epochs_dataloader \
    --log_name tvl_vittiny_tactile_encoder --shuffle_text --no_text_prompt \
    --replace_synonyms --num_workers 20 --use_not_contact \
    --tactile_model vit_tiny_patch16_224 --blr 3e-4 \
    --datasets_dir /your/data/dir
```

### Stage 1 Evaluation
```bash
# Touch-vision classification
python -m tvl_enc.tools.visualize_affinity \
    --checkpoint_path output/checkpoint.pth \
    --active_modality_names tactile vision \
    --tactile_model vit_tiny_patch16_224 \
    --datasets ssvtp hct --not_visualize --evaluate_all \
    --datasets_dir /your/data/dir
```

### SLURM Launch Scripts
```bash
sbatch experiments/stage2/claude_flextok/scripts/launch_stage2a.sh  # alignment
sbatch experiments/stage2/claude_flextok/scripts/launch_stage2b.sh  # reconstruction
sbatch experiments/stage2/claude_flextok/scripts/launch_multigpu.sh # 4-GPU alignment
```
Set `STAGE1_CKPT` and `DATASETS_DIR` env vars before running.

## Architecture

### Stage 1: TVL Encoder (`tvl_enc/`)

**`tvl_enc/tvl.py` — `TVL` class** is the core Stage-1 model:
- `clip` (OpenCLIP ViT-L-14, `datacomp_xl_s13b_b90k`): vision and text backbone, **frozen by default**
- `tactile_encoder` (TIMM ViT-Tiny/Small/Base): the only **trained** component in Stage 1
- Outputs normalized 768-d embeddings for each modality
- `forward(..., return_token_sequences=True, drop_cls_token=True)` returns per-patch token sequences `(B, N, 768)` instead of pooled embeddings — this is how Stage 2 consumes it

**`tvl_enc/tacvis.py`** defines:
- `TacVisDataset` (SSVTP dataset), `TacVisDatasetV2` (HCT dataset)
- `RGB_AUGMENTS`, `TAC_AUGMENTS` transform pipelines
- Normalization constants: `RGB_MEAN/STD`, `TAC_MEAN/STD`, `TAC_MEAN_BG/STD_BG`

The checkpoint `state_dict` saves only tactile encoder weights (CLIP weights are excluded and re-loaded from pretrained).

### Stage 2: FlexTok Register Tokens (`experiments/stage2/claude_flextok/`)

The central idea: frozen Stage-1 encoder token sequences → register token module → split into **shared tokens** (contrastively aligned across vision/tactile) and **private tokens** (preserve modality-specific info) → decoder reconstructs original images.

**Data flow:**
```
Input images (224×224)
  ↓ TVL.forward(return_token_sequences=True)  [frozen]
Patch token sequences (B, N, 768) per modality
  ↓ CrossModalAlignmentModel.forward()
    → RegisterTokenModule per modality
      → concat [patch_tokens | register_tokens]
      → FlexTransformer (causal mask among registers)
      → extract register_tokens only
      → apply nested dropout to shared registers
      → split: shared_tokens[:n_shared], private_tokens[n_shared:]
    → FSQEncoding on all_tokens (discrete regularizer)
    → shared_projectors → L2-normalized shared embedding
  ↓ Losses
    → CrossModalAlignmentLoss: contrastive on shared, preservation on private
    → FlowMatchingReconstructionDecoder (probe): decode all_tokens → (B, 3, 224, 224)
```

**Key classes:**

| File | Class | Role |
|------|-------|------|
| `models/register_tokens.py` | `RegisterTokenModule` | Per-modality register token encoder |
| `models/register_tokens.py` | `Registers1D` | Owns learnable register params + nested dropout |
| `models/register_tokens.py` | `RegisterTokenTransformer` | Transformer mixing input+register tokens |
| `models/cross_modal_alignment.py` | `CrossModalAlignmentModel` | Dual-modality wrapper, FSQ, projectors |
| `models/autoregressive_decoder.py` | `AutoregressiveDecoder` | Cross-attention decoder: registers→pixels (reconstruction stage) |
| `models/flow_matching_decoder.py` | `FlowMatchingReconstructionDecoder` | Rectified-flow probe decoder (alignment stage) |
| `pipeline/stage2_pipeline.py` | `Stage2Pipeline` | Wires frozen encoder + alignment model |
| `losses/alignment_loss.py` | `CrossModalAlignmentLoss` | CLIP-style contrastive + private preservation |
| `losses/flow_reconstruction_loss.py` | `FlowReconstructionLoss` | Flow-based probe loss (alignment stage) |
| `losses/reconstruction_loss.py` | `ReconstructionLoss` | Pixel-space L1/MSE loss with multi-scale supervision (reconstruction stage) |
| `build.py` | — | All dataset/model/loss builder functions (imported by `train.py`) |

**Nested dropout**: During training, trailing register tokens are randomly zeroed using independent power-of-two schedules applied **separately** to each group:
- Shared tokens: keep 1, 2, 4, … up to `n_shared` (e.g. schedule `[1,2,4,8]` for `n_shared=8`)
- Private tokens: keep 1, 2, 4, … up to `n_private` independently (e.g. `[1,2,4,8,16,24]` for `n_private=24`)

Each group samples its own `k_keep` per batch, enforcing coarse-to-fine ordering within both token sets. `RegisterTokenModule.forward()` returns `(shared_tokens, private_tokens, k_keep_shared, k_keep_private)`. **Nested dropout is automatically disabled in the reconstruction stage** — the decoder needs consistently fully-populated tokens to learn fine details. Controlled via `apply_nested_dropout` kwarg on `CrossModalAlignmentModel.forward()`.

**FSQ (Finite Scalar Quantization)**: All register tokens are quantized through `FSQEncoding` (default levels `[8,8,8,5,5,5]`) as a discrete regularizer, following FlexTok. Gradients pass through via straight-through estimator.

**Training stages (following FlexTok two-stage protocol):**
- `stage=alignment` **(default)**: trains `CrossModalAlignmentModel` (register modules + projectors) jointly with the flow-matching **probe decoder**. The probe acts as an information-bottleneck — it forces the tokens to retain pixel-reconstructable information while the contrastive loss aligns them cross-modally. Uses `model=alignment` and `losses=alignment`.
- `stage=reconstruction`: loads and freezes the trained alignment model, then trains a separate **autoregressive decoder** on top of the frozen tokens. Uses `model=reconstruction` and `losses=reconstruction`.

### Config System (Hydra)

`experiments/stage2/claude_flextok/configs/config.yaml` is the root config. It defaults to `model=alignment` and `losses=alignment`. Config groups are stage-specific:

**Losses config groups** (`configs/losses/`):
| File | Used for | Active losses |
|------|----------|---------------|
| `alignment.yaml` | Stage 2a | `losses.contrastive` (cross-modal) + `losses.probe` (flow-matching bottleneck) |
| `reconstruction.yaml` | Stage 2b | `losses.recon_decoder` (pixel-space L1 + multi-scale) |

**Model config groups** (`configs/model/`):
| File | Used for | Decoder section |
|------|----------|-----------------|
| `alignment.yaml` | Stage 2a | `model.probe_decoder` — arch params only, `_target_` injected as `FlowMatchingReconstructionDecoder` by `build.py` |
| `reconstruction.yaml` | Stage 2b | `model.recon_decoder` — includes `_target_: AutoregressiveDecoder` |

**`ReconstructionLoss` multi-scale supervision**: accepts `scale_weights` (list of `[scale_factor, weight]` pairs, e.g. `[[0.5, 0.5], [0.25, 0.25]]`) to add half- and quarter-resolution auxiliary losses alongside the full-resolution objective. Configured in the `pixel` section of `losses/reconstruction.yaml`.

### Checkpoint Format

Stage 2 checkpoints save keys: `cross_modality_encoder`, `alignment_model` (alias), `optimizer`, `scaler`, `epoch`, `full_config`, and `recon_decoder_{vision,tactile}` when decoders are present.

To load: `ckpt["cross_modality_encoder"]` → `model.cross_modality_encoder.load_state_dict(...)`.

## Dataset Structure

Both SSVTP and HCT datasets are expected under `--datasets_dir`:
- `<datasets_dir>/ssvtp/` — SSVTP subset
- `<datasets_dir>/hct/` — HCT subset (subdirs each with `contact.json`)

**Note**: The original dataset had a tactile/image folder swapping issue. Prefer the revised dataset at `yoorhim/TVL-revise` on HuggingFace.

## Debug Mode

Pass `debug_single_sample=true` (Hydra) to constrain the dataset to a single sample for overfit testing. Set `debug_dataset_mode=repeat_cached` (default) to repeat the sample in RAM and avoid disk IO.

## Modality String Keys

Modality names are string constants defined in `tvl_enc/tvl.py`:
```python
ModalityType.VISION   = "vision"
ModalityType.TACTILE  = "tactile"
ModalityType.TEXT     = "text"
```
`cross_modal_alignment.py` redefines `ModalityType` locally as a `SimpleNamespace` with the same values to avoid a circular import from `tvl_enc`.
