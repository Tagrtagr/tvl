"""
Quick smoke test for Stage 2 modules (Claude Code FlexTok approach).
Run: python experiments/stage2/claude_flextok/test_modules.py
"""

import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import torch
from models.register_tokens import RegisterTokenModule
from models.cross_modal_alignment import CrossModalAlignmentModel
from losses.alignment_loss import CrossModalAlignmentLoss
from losses.flow_matching import FlowMatchingAlignmentLoss
from models.reconstruction_decoder import ReconstructionDecoder
from losses.reconstruction_loss import ReconstructionLoss


def test_register_tokens():
    print("Testing RegisterTokenModule...")
    module = RegisterTokenModule(
        input_dim=768, hidden_dim=512, n_registers=16,
        n_shared=4, n_layers=2, n_heads=8,
        nested_dropout=True, nested_dropout_mode="power_of_two",
    )

    # Simulate pooled encoder features (B=4, 1 token, 768-dim)
    x = torch.randn(4, 1, 768)

    # Training mode (nested dropout active)
    module.train()
    shared, private, k_keep = module(x)
    print(f"  [train] shared: {shared.shape}, private: {private.shape}, k_keep: {k_keep}")
    assert shared.shape == (4, 4, 512), f"Expected (4, 4, 512), got {shared.shape}"
    assert private.shape == (4, 12, 512), f"Expected (4, 12, 512), got {private.shape}"

    # Eval mode (no dropout)
    module.eval()
    shared, private, k_keep = module(x)
    print(f"  [eval]  shared: {shared.shape}, private: {private.shape}, k_keep: {k_keep}")
    assert k_keep == 16
    print("  PASSED")


def test_cross_modal_alignment():
    print("Testing CrossModalAlignmentModel...")
    model = CrossModalAlignmentModel(
        modality_configs={
            "vision": {"input_dim": 768, "feature_type": "pooled"},
            "tactile": {"input_dim": 768, "feature_type": "pooled"},
        },
        hidden_dim=256, n_registers=16, n_shared=4,
        n_layers=2, n_heads=4,
    )

    features = {
        "vision": torch.randn(4, 768),
        "tactile": torch.randn(4, 768),
    }

    model.train()
    output = model(features)

    assert "vision_shared" in output
    assert "tactile_shared" in output
    assert "vision_private" in output
    assert "tactile_private" in output
    assert "logit_scale" in output

    print(f"  vision_shared: {output['vision_shared'].shape}")
    print(f"  tactile_shared: {output['tactile_shared'].shape}")
    print(f"  logit_scale: {output['logit_scale'].item():.4f}")

    # Check L2 normalization
    norms = output["vision_shared"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Shared embeddings should be L2-normalized"
    print("  PASSED")


def test_contrastive_loss():
    print("Testing CrossModalAlignmentLoss...")
    loss_fn = CrossModalAlignmentLoss(
        modality_names=["vision", "tactile"],
        contrastive_weight=1.0,
        preservation_weight=0.5,
    )

    alignment_output = {
        "vision_shared": torch.randn(8, 256),
        "tactile_shared": torch.randn(8, 256),
        "vision_private": torch.randn(8, 12, 256),
        "tactile_private": torch.randn(8, 12, 256),
        "logit_scale": torch.tensor(14.3),
    }
    # Normalize shared
    alignment_output["vision_shared"] = torch.nn.functional.normalize(alignment_output["vision_shared"], dim=-1)
    alignment_output["tactile_shared"] = torch.nn.functional.normalize(alignment_output["tactile_shared"], dim=-1)

    frozen_features = {
        "vision": torch.randn(8, 768),
        "tactile": torch.randn(8, 768),
    }

    # Preservation projectors to match dims (256 -> 768)
    preservation_projectors = {
        "vision": torch.nn.Linear(256, 768),
        "tactile": torch.nn.Linear(256, 768),
    }

    losses = loss_fn(alignment_output, frozen_features=frozen_features, preservation_projectors=preservation_projectors)

    print(f"  total_loss: {losses['total_loss'].item():.4f}")
    print(f"  contrastive_avg: {losses['contrastive_avg'].item():.4f}")
    print(f"  preservation_avg: {losses['preservation_avg'].item():.4f}")
    print(f"  acc1_avg: {losses['acc1_avg'].item():.1f}%")
    print("  PASSED")


def test_flow_matching():
    print("Testing FlowMatchingAlignmentLoss...")
    flow_loss = FlowMatchingAlignmentLoss(
        modality_names=["vision", "tactile"],
        embed_dim=256, hidden_dim=512, n_layers=2,
        bidirectional=True,
    )

    alignment_output = {
        "vision_shared": torch.nn.functional.normalize(torch.randn(8, 256), dim=-1),
        "tactile_shared": torch.nn.functional.normalize(torch.randn(8, 256), dim=-1),
        "logit_scale": torch.tensor(14.3),
    }

    losses = flow_loss(alignment_output)
    print(f"  flow_total: {losses['flow_total'].item():.4f}")
    print(f"  flow_avg: {losses['flow_avg'].item():.4f}")

    # Test transport
    x = torch.randn(4, 256)
    transported = flow_loss.transport(x, "vision", "tactile", n_steps=5)
    print(f"  transport shape: {transported.shape}")
    assert transported.shape == x.shape
    print("  PASSED")


def test_reconstruction_decoder():
    print("Testing ReconstructionDecoder...")
    decoder = ReconstructionDecoder(
        n_registers=16, hidden_dim=256, base_channels=128,
        n_decoder_layers=1, n_heads=4,
    )

    # Simulate all register tokens (shared + private)
    tokens = torch.randn(4, 16, 256)
    recon = decoder(tokens)
    print(f"  input: {tokens.shape}")
    print(f"  output: {recon.shape}")
    assert recon.shape == (4, 3, 224, 224), f"Expected (4, 3, 224, 224), got {recon.shape}"

    # Check gradients flow
    recon.sum().backward()
    has_grad = any(p.grad is not None for p in decoder.parameters())
    assert has_grad, "No gradients in decoder!"
    print(f"  params: {sum(p.numel() for p in decoder.parameters()):,}")
    print("  PASSED")


def test_reconstruction_loss():
    print("Testing ReconstructionLoss...")
    loss_fn = ReconstructionLoss(loss_type="mse")

    recons = {"vision": torch.randn(4, 3, 224, 224), "tactile": torch.randn(4, 3, 224, 224)}
    targets = {"vision": torch.randn(4, 3, 224, 224), "tactile": torch.randn(4, 3, 224, 224)}

    losses = loss_fn(recons, targets)
    print(f"  recon_pixel_vision: {losses['recon_pixel_vision'].item():.4f}")
    print(f"  recon_pixel_tactile: {losses['recon_pixel_tactile'].item():.4f}")
    print(f"  recon_total: {losses['recon_total'].item():.4f}")
    assert losses["recon_total"].item() > 0
    print("  PASSED")


def test_reconstruction_backward():
    print("Testing reconstruction end-to-end backward...")
    model = CrossModalAlignmentModel(
        modality_configs={
            "vision": {"input_dim": 768, "feature_type": "pooled"},
            "tactile": {"input_dim": 768, "feature_type": "pooled"},
        },
        hidden_dim=256, n_registers=16, n_shared=4,
        n_layers=2, n_heads=4,
    )
    decoder_vision = ReconstructionDecoder(
        n_registers=16, hidden_dim=256, base_channels=64,
        n_decoder_layers=1, n_heads=4,
    )
    decoder_tactile = ReconstructionDecoder(
        n_registers=16, hidden_dim=256, base_channels=64,
        n_decoder_layers=1, n_heads=4,
    )
    recon_loss_fn = ReconstructionLoss(loss_type="mse")

    # Simulate frozen features and raw images
    features = {"vision": torch.randn(2, 768), "tactile": torch.randn(2, 768)}
    raw_images = {"vision": torch.randn(2, 3, 224, 224), "tactile": torch.randn(2, 3, 224, 224)}

    model.train()
    output = model(features)

    # Decode
    recon_vision = decoder_vision(output["vision_all_tokens"])
    recon_tactile = decoder_tactile(output["tactile_all_tokens"])
    recons = {"vision": recon_vision, "tactile": recon_tactile}

    losses = recon_loss_fn(recons, raw_images)
    loss = losses["recon_total"]
    loss.backward()

    # Check gradients flow through decoder AND alignment model
    decoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder_vision.parameters())
    model_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert decoder_has_grad, "No gradients in decoder!"
    assert model_has_grad, "No gradients in alignment model!"
    print(f"  recon_loss: {loss.item():.4f}")
    print(f"  decoder grads: OK")
    print(f"  alignment model grads: OK")
    print("  PASSED")


def test_backward():
    print("Testing backward pass (end-to-end)...")
    model = CrossModalAlignmentModel(
        modality_configs={
            "vision": {"input_dim": 768, "feature_type": "pooled"},
            "tactile": {"input_dim": 768, "feature_type": "pooled"},
        },
        hidden_dim=256, n_registers=16, n_shared=4,
        n_layers=2, n_heads=4,
    )
    loss_fn = CrossModalAlignmentLoss(modality_names=["vision", "tactile"])

    features = {
        "vision": torch.randn(4, 768),
        "tactile": torch.randn(4, 768),
    }

    model.train()
    output = model(features)
    losses = loss_fn(
        output, frozen_features=features,
        preservation_projectors=model.preservation_projectors,
    )
    loss = losses["total_loss"]
    loss.backward()

    # Check gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed!"
    print(f"  loss: {loss.item():.4f}")
    print(f"  gradients: OK")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 2 Module Tests")
    print("=" * 60)
    test_register_tokens()
    test_cross_modal_alignment()
    test_contrastive_loss()
    test_flow_matching()
    test_backward()
    test_reconstruction_decoder()
    test_reconstruction_loss()
    test_reconstruction_backward()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
