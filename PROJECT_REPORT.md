# TVL Stage 2: Cross-Modality Alignment — Project Report

**Project:** Touch-Vision-Language (TVL) Cross-Modality Alignment with FlexTok-Inspired Register Tokens
**Repository:** `Tagrtagr/tvl`
**Branch:** `claude/cross-modality-alignment-xo86K`
**Date:** March 12, 2026
**Duration:** Feb 6, 2026 — Present (~5 weeks)

---

## 1. Project Overview

### 1.1 Background

TVL (Touch, Vision, and Language) is a multimodal alignment framework from the paper *"A Touch, Vision, and Language Dataset for Multimodal Alignment"* (Fu et al., 2024). The original codebase provides:

- **Stage 1** (`tvl_enc/`): Tactile encoder pre-training aligned with CLIP vision+text encoders via contrastive loss
- **Stage 3** (`tvl_llama/`): TVL-LLaMA — a multimodal LLM adapter built on LLaMA-2-7b

This project adds **Stage 2**: a cross-modality alignment layer using **FlexTok-inspired register tokens** that sits between the frozen Stage 1 encoders and downstream tasks, learning a shared bottleneck representation across touch and vision.

### 1.2 Motivation

Stage 1 produces aligned embeddings via contrastive learning, but:
- The alignment is coarse (single pooled vector per modality)
- There is no mechanism for preserving modality-specific details alongside shared information
- There is no hierarchical/progressive representation (coarse-to-fine)

Stage 2 addresses this by learning **register tokens** that compress encoder features into a fixed-length bottleneck with explicit shared (cross-modal) and private (modality-specific) partitions, trained with **nested dropout** for coarse-to-fine ordering.

### 1.3 Key Inspirations

| Paper | Idea Borrowed |
|---|---|
| **FlexTok** (Bachmann et al., 2025) | Register tokens, nested dropout, variable-length representations |
| **OAT** (Ordered Action Tokenization) | Prefix reconstruction, coarse-to-fine decoding, token type embeddings |
| **CLIP** (Radford et al., 2021) | Symmetric contrastive alignment loss |

---

## 2. Architecture

### 2.1 System Diagram

```
                          Stage 1 (Frozen)                    Stage 2 (Learned)
                    ┌─────────────────────┐           ┌──────────────────────────┐
  Vision Image ───► │  OpenCLIP ViT-L-14  │──768d──►  │  RegisterTokenModule     │
                    └─────────────────────┘           │  (4-layer Transformer)   │──► [S₁..S₈ | P₉..P₃₂]
                                                      │  + Causal Attention      │        │         │
  Tactile Image ──► ┌─────────────────────┐           │  + Nested Dropout        │    Shared    Private
                    │  ViT-Tiny (aligned)  │──768d──►  │  + Token Type Embeddings │      │         │
                    └─────────────────────┘           └──────────────────────────┘      │         │
                                                                                        ▼         ▼
                                                                              Contrastive    Preservation
                                                                                 Loss          Loss
                                                                                        │
                                                                    ┌────────────────────┘
                                                                    ▼
                                                      ┌──────────────────────────┐
                                                      │  Autoregressive Decoder  │──► Reconstructed Image
                                                      │  (Transformer + ConvUp)  │    (224×224)
                                                      └──────────────────────────┘
```

### 2.2 Core Components

#### Register Token Module (`models/register_tokens.py`)
- Projects frozen encoder features (768d) → hidden_dim (512d)
- Prepends learnable register tokens [n_registers=32] to projected features
- Processes through 4-layer causal Transformer (enforces coarse-to-fine ordering)
- Outputs only register tokens (discards input features → bottleneck)
- **Nested dropout**: Randomly masks suffix tokens during training (power-of-two sampling)
- **Token type embeddings**: Distinguishes shared vs private tokens

#### Cross-Modal Alignment Model (`models/cross_modal_alignment.py`)
- Separate RegisterTokenModule per modality (vision, tactile)
- Shared projection heads pool shared tokens → single embedding for contrastive loss
- Preservation projectors map private tokens back to encoder dimension

#### Autoregressive Decoder (`models/autoregressive_decoder.py`)
- Transformer decoder with spatial queries cross-attending to register tokens
- ConvTranspose upsampling: hidden → 14×14 → 28×28 → 56×56 → 112×112 → 224×224
- Supports prefix decoding: variable number of register tokens → progressive quality

### 2.3 Loss Functions

| Loss | Purpose | Weight |
|---|---|---|
| **Contrastive** (alignment_loss.py) | Align shared tokens across modalities (CLIP-style symmetric) | 1.0 |
| **Preservation** (alignment_loss.py) | MSE between private tokens and frozen encoder features | 0.5 |
| **Pixel Reconstruction** (reconstruction_loss.py) | MSE/L1 between decoded and original images | 1.0 |
| **Prefix Reconstruction** (reconstruction_loss.py) | OAT-style: reconstruct from each prefix length k=1,2,4,...,32 | 1.0 |
| **Flow Matching** (flow_matching.py) | Optional continuous flow between modalities | 0.5 |

### 2.4 Training Pipeline

**Two-stage FlexTok workflow:**

| Stage | What Trains | What's Frozen | Goal |
|---|---|---|---|
| **Stage 2a** | Register tokens + alignment model | Stage 1 encoders | Learn cross-modal bottleneck |
| **Stage 2b** | Autoregressive decoder only | Stage 1 encoders + alignment model | Learn to reconstruct from bottleneck |

---

## 3. Implementation Details

### 3.1 Files Created

```
experiments/stage2/claude_flextok/
├── configs/
│   └── default.yaml                    # Full configuration template
├── models/
│   ├── __init__.py
│   ├── register_tokens.py              # Register token module with nested dropout
│   ├── cross_modal_alignment.py        # Cross-modal alignment wrapper
│   ├── autoregressive_decoder.py       # Transformer decoder for reconstruction
│   └── reconstruction_decoder.py       # CNN decoder (earlier variant)
├── losses/
│   ├── __init__.py
│   ├── alignment_loss.py               # Contrastive + preservation loss
│   ├── flow_matching.py                # Flow matching loss (optional)
│   └── reconstruction_loss.py          # Pixel + prefix reconstruction loss
├── scripts/
│   ├── launch_stage2a.sh               # SLURM launcher for Stage 2a
│   ├── launch_stage2b.sh               # SLURM launcher for Stage 2b
│   └── launch_multigpu.sh             # Multi-GPU launcher
├── train.py                            # Main training script (DDP-aware)
├── visualize.py                        # General visualization script
├── visualize_prefix_recon.py           # OAT-style prefix reconstruction viz
├── run_visualize.sh                    # SLURM script for visualization
├── strip_checkpoint.py                 # Utility to remove optimizer state from checkpoints
├── sweep_registers.py                  # Hyperparameter sweep over n_registers
├── test_modules.py                     # Unit tests for all modules
└── README.md                           # Stage 2 documentation
```

### 3.2 Configuration

```yaml
model:
  hidden_dim: 512          # Register token dimension
  n_registers: 32          # Total register tokens
  n_shared: 8              # Tokens for cross-modal alignment
  n_layers: 4              # Transformer depth per modality
  n_heads: 8               # Attention heads
  nested_dropout: true     # Enable coarse-to-fine training
  nested_dropout_mode: "power_of_two"  # k ∈ {1, 2, 4, 8, 16, 32}

loss:
  loss_type: "contrastive"
  contrastive_weight: 1.0
  preservation_weight: 0.5

training:
  epochs: 100
  batch_size: 256 (2a) / 32 (2b, with accum_iter=8)
  base_lr: 3e-4
  weight_decay: 0.05
  warmup_epochs: 10

stage1:
  tactile_model: "vit_tiny_patch16_224"
  clip_vision_model: "ViT-L-14"
  checkpoint: /viscam/u/taarush/tvl_enc_vittiny.pth
```

### 3.3 Infrastructure

- **Cluster**: VISCAM partition (Stanford)
- **GPU**: Single A100 (Stage 2a/2b), multi-GPU supported via DDP
- **Dataset**: SSVTP (HCT not available on cluster)
- **Mixed precision**: float16 for decoder, float32 for alignment
- **SLURM**: Custom launch scripts with account/partition/chdir handling

---

## 4. Development Timeline

### Phase 1: Scaffolding (Feb 6–10)
| Date | Commits | Work |
|---|---|---|
| Feb 6 | 1af1664 | Initial Stage 2 experiment scaffold |
| Feb 7 | 0cac552 | Full FlexTok-inspired implementation (register tokens, alignment model, losses) |
| Feb 10 | 0c7d907, 7f870a2 | Namespace under `claude_flextok/`, remove timm dependency from losses |

### Phase 2: Core Components (Feb 17)
| Date | Commits | Work |
|---|---|---|
| Feb 17 | 2028067 | Make tensorboard import optional |
| Feb 17 | 9000507 | Fix tactile_dim detection (use num_classes) |
| Feb 17 | 7bb2de6 | Fix DDP wrapper (unwrap model for inner attributes) |
| Feb 17 | 49ca635 | Add FlexTok reconstruction decoder |
| Feb 17 | 12560f7 | Fix OOM: bottleneck in decoder, reduce base_channels |

### Phase 3: Visualization & PR (Feb 26)
| Date | Commits | Work |
|---|---|---|
| Feb 26 | af9cafb | Comprehensive visualization script |
| Feb 26 | 4fdca2e | **PR #2 merged** into main |

### Phase 4: Advanced Features (Mar 3–4)
| Date | Commits | Work |
|---|---|---|
| Mar 3 | f7cbf53 | Register token sweep, OAT-style prefix reconstruction, token type embeddings |
| Mar 4 | 4432ab1 | Autoregressive decoder, two-stage training split (2a/2b) |
| Mar 4 | bb63d6a | Fix single-GPU SLURM (skip DDP when SLURM_NTASKS=1) |

### Phase 5: Cluster Deployment (Mar 9)
| Date | Commits | Work |
|---|---|---|
| Mar 9 | 76fb920 | SLURM launch scripts for Stage 2a/2b |
| Mar 9 | ede6b3b–940f9b0 | 6 commits fixing SLURM paths, dataset availability, account settings |
| Mar 9 | c9b5436 | **Critical fix**: nested dropout was zeroing private tokens |

### Phase 6: Bug Fixes & Training (Mar 10)
| Date | Commits | Work |
|---|---|---|
| Mar 10 | 5d4f386 | Fix ssvtp dataset path (nested one level deeper) |
| Mar 10 | 2e975e6 | Fix config key mismatches, wrong loss weight, squeeze bug |
| Mar 10 | 454eed5 | Fix forward_prefix to mask suffix tokens from self-attention |
| Mar 10 | 99c2a21 | Address CodeRabbit review comments |
| Mar 10 | d046e95 | Fix config merge logic, scontrol error handling, device mismatch |
| Mar 10 | dbaa14e | Fix DDP, checkpoint validation, SLURM, decoder loading |
| Mar 10 | bbb86b8 | Fix reproducibility, double-scaling, validation, DDP, config bugs |
| Mar 10 | 9f6411b–87c5043 | Fix SBATCH headers and working directory paths |
| Mar 10 | e6787cc | Fix Stage 2b OOM: batch_size=32 with accum_iter=8 |

### Phase 7: Visualization & Analysis (Mar 11)
| Date | Commits | Work |
|---|---|---|
| Mar 11 | d1c0b9e–1cb5bd2 | 5 commits fixing OOM in visualization (mmap, fp16, strip checkpoint) |
| Mar 11 | 3b1e486 | SBATCH script for prefix reconstruction visualization |
| Mar 11 | 3133261 | Fix dtype mismatch between alignment (fp32) and decoder (fp16) |
| Mar 11 | 48e160c | **Fix**: add missing `--stage1_checkpoint` to visualization |

**Total: 42 commits across 5 weeks**

---

## 5. Training Runs

### 5.1 Stage 2a: Alignment Training
- **Script**: `scripts/launch_stage2a.sh`
- **Output**: `experiments/stage2/runs/stage2a_align/`
- **Status**: Completed
- **Config**: n_registers=32, n_shared=8, contrastive + preservation loss

### 5.2 Stage 2b: Reconstruction Training
- **Script**: `scripts/launch_stage2b.sh`
- **Output**: `experiments/stage2/runs/stage2b_recon/`
- **Checkpoint**: `checkpoint_best.pth`
- **Status**: Completed
- **Config**: Frozen alignment model from 2a, autoregressive decoder, prefix reconstruction loss

---

## 6. Current Results & Analysis

### 6.1 Prefix Reconstruction MSE (n_registers=32)

Results from `visualize_prefix_recon.py` on the trained Stage 2b checkpoint:

| Prefix Length K | Vision MSE | Tactile MSE |
|---|---|---|
| 1 | 0.2352 | 0.8416 |
| 2 | 0.2345 | 0.8605 |
| 4 | 0.2366 | 0.7689 |
| 8 | 0.2398 | 0.7697 |
| 16 | 0.2362 | 0.7307 |
| 32 | 0.2389 | 0.7588 |

### 6.2 Qualitative Observations

**Vision reconstructions**: All prefix lengths (k=1 through k=32) produce near-identical gray-brown blurs resembling the dataset mean. No structure or texture is recovered. MSE is flat (~0.235), confirming tokens carry no progressive information.

**Tactile reconstructions**: Slightly better — rough color blobs captured at k=1, but k=2 through k=32 show minimal improvement. MSE decreases somewhat (0.84 → 0.73) but is not monotonic (increases at k=32).

### 6.3 Diagnosis

The visualization was run **without `--stage1_checkpoint`**, meaning:
- The tactile encoder had **random weights** (ViT-Tiny not loaded from Stage 1)
- The alignment model received meaningless tactile features
- Vision features (from pretrained CLIP) were partially correct but the alignment model was trained with Stage 1 features

**This invalidates the current results.** The fix has been committed (48e160c) — rerunning with `--stage1_checkpoint /viscam/u/taarush/tvl_enc_vittiny.pth` is required.

### 6.4 Expected Behavior (If Correct)

With proper Stage 1 features, we expect:
- MSE should **decrease monotonically** as K increases (coarse-to-fine)
- k=1 should capture global structure (mean color, overall shape)
- k=32 should capture fine details (texture, edges)
- Vision should reconstruct better than tactile (CLIP features are stronger)

---

## 7. Known Issues & Next Steps

### 7.1 Immediate Action Items

| Priority | Issue | Status |
|---|---|---|
| **P0** | Rerun visualization with `--stage1_checkpoint` | Fixed in code, needs rerun |
| **P1** | Validate that training used correct Stage 1 checkpoint | Confirmed (launch_stage2b.sh uses tvl_enc_vittiny.pth) |
| **P1** | Check training loss curves (tensorboard logs) | Not yet reviewed |

### 7.2 If Results Are Still Poor After Fix

If the corrected visualization still shows flat MSE / no coarse-to-fine ordering:

1. **Check training convergence**: Review tensorboard logs for loss curves
2. **Decoder capacity**: The autoregressive decoder may need more layers/channels
3. **Prefix loss weighting**: The prefix reconstruction loss may need higher weight relative to alignment loss
4. **Register token count**: 32 may be too many — try 8 or 16
5. **Training duration**: Stage 2b may need more epochs
6. **Nested dropout schedule**: Consider annealing from aggressive dropout to mild

### 7.3 Future Work

- **Register token sweep**: `sweep_registers.py` is ready to sweep n_registers ∈ {4, 8, 16, 32, 64}
- **Flow matching**: Optional loss implemented but not yet tested
- **Text modality**: RegisterTokenModule supports text but not yet integrated into training
- **Downstream evaluation**: Use register tokens as input to TVL-LLaMA (Stage 3)
- **Cross-modal generation**: Use flow matching to generate tactile from vision (and vice versa)

---

## 8. Summary Statistics

| Metric | Value |
|---|---|
| **Total commits** | 42 (Stage 2 work) |
| **Python files created** | 14 |
| **Shell scripts created** | 5 |
| **Config files** | 1 |
| **Model components** | 4 (register tokens, alignment, AR decoder, CNN decoder) |
| **Loss functions** | 3 (contrastive+preservation, flow matching, reconstruction) |
| **Training stages** | 2 (2a alignment, 2b reconstruction) |
| **Visualization tools** | 3 (general, prefix recon, sweep) |
| **Utility scripts** | 2 (strip checkpoint, test modules) |
| **Development duration** | ~5 weeks (Feb 6 — Mar 12, 2026) |
| **PRs merged** | 1 (PR #2) |
| **Critical bugs fixed** | ~15 (DDP, OOM, SLURM, dtype, dropout, config) |

---

*Report generated March 12, 2026*
