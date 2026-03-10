#!/usr/bin/env python3
"""
Register Token Count Sweep for FlexTok Stage 2.

Launches training runs with different numbers of register tokens (8, 16, 32, 64)
to find the optimal token budget. Each run trains with reconstruction enabled,
following the OAT insight that different token counts capture different levels
of information granularity.

Usage:
    # Run all sweeps sequentially (single GPU):
    python experiments/stage2/claude_flextok/sweep_registers.py \
        --stage1_checkpoint /path/to/stage1.pth \
        --datasets_dir /path/to/datasets \
        --output_root ./output/sweep_registers

    # Run a specific config only:
    python experiments/stage2/claude_flextok/sweep_registers.py \
        --stage1_checkpoint /path/to/stage1.pth \
        --datasets_dir /path/to/datasets \
        --output_root ./output/sweep_registers \
        --n_registers 8 16

    # Generate comparison after sweeps are done:
    python experiments/stage2/claude_flextok/sweep_registers.py \
        --output_root ./output/sweep_registers \
        --compare_only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Sweep configurations ──
# n_shared is kept at 25% of n_registers (following the 8/32 default ratio)
SWEEP_CONFIGS = {
    8:  {"n_registers": 8,  "n_shared": 2},
    16: {"n_registers": 16, "n_shared": 4},
    32: {"n_registers": 32, "n_shared": 8},
    64: {"n_registers": 64, "n_shared": 16},
}


def parse_log(log_path):
    """Parse a log.txt file (JSON lines) into a list of dicts."""
    entries = []
    if not os.path.exists(log_path):
        return entries
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def run_sweep(args):
    """Launch training runs for each register token count."""
    train_script = os.path.join(os.path.dirname(__file__), "train.py")

    for n_reg in args.n_registers:
        config = SWEEP_CONFIGS[n_reg]
        run_name = f"reg{n_reg}"
        output_dir = os.path.join(args.output_root, run_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Sweep: n_registers={n_reg}, n_shared={config['n_shared']}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, train_script,
            "--n_registers", str(config["n_registers"]),
            "--n_shared", str(config["n_shared"]),
            "--use_reconstruction",
            "--reconstruction_weight", str(args.reconstruction_weight),
            "--output_dir", output_dir,
            "--log_name", run_name,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
        ]

        if args.stage1_checkpoint:
            cmd += ["--stage1_checkpoint", args.stage1_checkpoint]
        if args.datasets_dir:
            cmd += ["--datasets_dir", args.datasets_dir]

        # Log the command
        cmd_str = " ".join(cmd)
        print(f"Running: {cmd_str}\n")
        with open(os.path.join(output_dir, "command.txt"), "w") as f:
            f.write(cmd_str + "\n")

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=os.path.dirname(train_script))
            if result.returncode != 0:
                print(f"WARNING: Training with n_registers={n_reg} exited with code {result.returncode}")
        else:
            print("(dry run — skipping)")


def compare_sweeps(output_root, output_dir=None):
    """Compare results across all sweep runs and generate plots."""
    if output_dir is None:
        output_dir = os.path.join(output_root, "comparison")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Collect results from each run
    results = {}
    for n_reg in sorted(SWEEP_CONFIGS.keys()):
        log_path = os.path.join(output_root, f"reg{n_reg}", "log.txt")
        entries = parse_log(log_path)
        if entries:
            results[n_reg] = entries
            print(f"  reg{n_reg}: {len(entries)} epochs logged")
        else:
            alt_log = os.path.join(output_root, f"reg{n_reg}", f"reg{n_reg}", "log.txt")
            entries = parse_log(alt_log)
            if entries:
                results[n_reg] = entries
                print(f"  reg{n_reg}: {len(entries)} epochs logged (nested dir)")

    if not results:
        print("No sweep results found. Run training first.")
        return

    # ── Plot 1: Accuracy vs n_registers (bar chart) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_regs = sorted(results.keys())
    best_acc1 = []
    final_recon_loss = []
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for n_reg in n_regs:
        entries = results[n_reg]
        best = max(e.get("val_acc1", 0) for e in entries)
        best_acc1.append(best)
        last = entries[-1]
        recon = last.get("val_recon_total", last.get("val_recon_pixel_avg", 0))
        final_recon_loss.append(recon)

    # Bar chart: best accuracy
    ax = axes[0]
    bars = ax.bar([str(n) for n in n_regs], best_acc1,
                  color=colors[:len(n_regs)], edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, best_acc1, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Number of Register Tokens", fontsize=12)
    ax.set_ylabel("Best Val Acc@1 (%)", fontsize=12)
    ax.set_title("Best Retrieval Accuracy vs Token Count", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(best_acc1) + 10)
    ax.grid(True, alpha=0.3, axis="y")

    # Bar chart: final reconstruction loss
    ax = axes[1]
    bars = ax.bar([str(n) for n in n_regs], final_recon_loss,
                  color=colors[:len(n_regs)], edgecolor="black", linewidth=0.5)
    for bar, loss_val in zip(bars, final_recon_loss, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                f"{loss_val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of Register Tokens", fontsize=12)
    ax.set_ylabel("Final Recon Loss (MSE)", fontsize=12)
    ax.set_title("Reconstruction Loss vs Token Count", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Register Token Count Sweep", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "sweep_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Plot 2: Training curves overlay ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves Across Register Token Counts", fontsize=15, fontweight="bold")

    for i, n_reg in enumerate(n_regs):
        entries = results[n_reg]
        epochs = [e["epoch"] for e in entries]
        color = colors[i % len(colors)]
        label = f"reg={n_reg}"

        # Val accuracy
        val_acc = [e.get("val_acc1", 0) for e in entries]
        axes[0, 0].plot(epochs, val_acc, color=color, label=label, linewidth=2)

        # Train accuracy
        train_acc = [e.get("train_acc1", e.get("train_acc1_avg", 0)) for e in entries]
        axes[0, 1].plot(epochs, train_acc, color=color, label=label, linewidth=2)

        # Total loss
        val_loss = [e.get("val_loss", 0) for e in entries]
        axes[1, 0].plot(epochs, val_loss, color=color, label=label, linewidth=2)

        # Recon loss
        recon_loss = [e.get("val_recon_total", e.get("val_recon_pixel_avg", 0)) for e in entries]
        axes[1, 1].plot(epochs, recon_loss, color=color, label=label, linewidth=2)

    for ax, title, ylabel in [
        (axes[0, 0], "Validation Accuracy", "Acc@1 (%)"),
        (axes[0, 1], "Training Accuracy", "Acc@1 (%)"),
        (axes[1, 0], "Validation Loss", "Loss"),
        (axes[1, 1], "Reconstruction Loss", "MSE"),
    ]:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "sweep_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print(f"  {'n_registers':>12} | {'n_shared':>8} | {'Best Acc@1':>10} | {'Final Recon':>12} | {'Epochs':>6}")
    print("-" * 70)
    for n_reg, acc, recon in zip(n_regs, best_acc1, final_recon_loss, strict=True):
        cfg = SWEEP_CONFIGS[n_reg]
        n_epochs = len(results[n_reg])
        print(f"  {n_reg:>12} | {cfg['n_shared']:>8} | {acc:>9.1f}% | {recon:>12.4f} | {n_epochs:>6}")
    print("=" * 70)

    # Save results JSON
    summary = {
        str(n_reg): {
            "n_registers": n_reg,
            "n_shared": SWEEP_CONFIGS[n_reg]["n_shared"],
            "best_acc1": acc,
            "final_recon_loss": recon,
            "n_epochs": len(results[n_reg]),
        }
        for n_reg, acc, recon in zip(n_regs, best_acc1, final_recon_loss)
    }
    json_path = os.path.join(output_dir, "sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser("Register Token Sweep")

    # Sweep settings
    parser.add_argument("--n_registers", type=int, nargs="+", default=[8, 16, 32, 64],
                        help="Register token counts to sweep")
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)

    # Paths
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--datasets_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="./output/sweep_registers")

    # Modes
    parser.add_argument("--compare_only", action="store_true",
                        help="Skip training, only generate comparison plots from existing logs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")

    args = parser.parse_args()

    if not args.compare_only:
        run_sweep(args)

    print("\nGenerating comparison plots...")
    compare_sweeps(args.output_root)


if __name__ == "__main__":
    main()
