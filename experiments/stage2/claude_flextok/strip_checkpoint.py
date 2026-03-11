#!/usr/bin/env python3
"""Strip optimizer/scaler from a PyTorch checkpoint to reduce file size.

This script processes the checkpoint zip file entry-by-entry, copying only
the keys needed for inference (args, alignment_model, recon_decoder_*).
It avoids loading the full checkpoint into memory, so it works on machines
with limited RAM.

Usage:
    python experiments/stage2/claude_flextok/strip_checkpoint.py \
        experiments/stage2/runs/stage2b_recon/stage2b_recon/checkpoint_best.pth
"""
import sys
import os
import pickle
import io
import zipfile
import shutil
import tempfile
import struct


def get_checkpoint_keys(path):
    """Peek at the top-level keys of a torch checkpoint without loading tensors."""
    # PyTorch checkpoints are zip files with a data.pkl and tensor files.
    # We can read data.pkl and partially unpickle to find top-level keys.
    with zipfile.ZipFile(path, "r") as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            return None
        with zf.open(pkl_names[0]) as f:
            # Read raw pickle bytes
            data = f.read()

    # The top-level object is a dict. We can use a restricted unpickler
    # that stubs out tensor reconstruction to just get the keys.
    class _StubStorage:
        def __init__(self, *args, **kwargs):
            pass

    class _KeyOnlyUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            # Return a stub instead of loading actual tensor data
            return _StubStorage()

        def find_class(self, module, name):
            # Allow safe built-in types; stub everything else
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return lambda *args, **kwargs: None
            if module == "torch" and name in ("FloatStorage", "HalfStorage",
                                               "LongStorage", "IntStorage",
                                               "ShortStorage", "DoubleStorage",
                                               "BFloat16Storage", "ByteStorage",
                                               "CharStorage", "BoolStorage",
                                               "ComplexFloatStorage",
                                               "ComplexDoubleStorage",
                                               "QInt8Storage", "QInt32Storage",
                                               "QUInt8Storage", "QUInt4x2Storage",
                                               "QUInt2x4Storage",
                                               "storage", "UntypedStorage",
                                               "TypedStorage"):
                return _StubStorage
            if module == "torch._utils" and name == "_rebuild_parameter":
                return lambda *args, **kwargs: None
            if module == "torch._utils" and name == "_rebuild_parameter_with_state":
                return lambda *args, **kwargs: None
            if module == "torch" and name in ("device", "Size"):
                return lambda *args, **kwargs: None
            # For everything else, try the default
            try:
                return super().find_class(module, name)
            except Exception:
                return lambda *args, **kwargs: None

    try:
        result = _KeyOnlyUnpickler(io.BytesIO(data)).load()
        if isinstance(result, dict):
            return list(result.keys())
    except Exception as e:
        print(f"Warning: Could not peek at keys: {e}")
    return None


def strip_checkpoint(input_path, output_path=None, keep_keys=None):
    """Load checkpoint, remove optimizer/scaler, re-save.

    Uses torch.load but immediately frees heavy keys before any
    downstream processing.
    """
    import torch
    import gc

    if output_path is None:
        output_path = input_path.replace(".pth", ".stripped.pth")

    if os.path.exists(output_path):
        print(f"Stripped checkpoint already exists: {output_path}")
        return output_path

    remove_keys = {"optimizer", "scaler"}
    if keep_keys:
        # Only keep specified keys
        pass

    print(f"Input:  {input_path}")
    print(f"Input size: {os.path.getsize(input_path) / 1e9:.2f} GB")

    # First, peek at the keys to show what we'll remove
    keys = get_checkpoint_keys(input_path)
    if keys:
        print(f"Checkpoint keys: {keys}")
        print(f"Will remove: {[k for k in keys if k in remove_keys]}")

    print(f"\nLoading checkpoint...")
    ckpt = torch.load(input_path, map_location="cpu")

    for key in list(ckpt.keys()):
        if key in remove_keys:
            print(f"  Deleting '{key}'...")
            del ckpt[key]
            gc.collect()

    print(f"Saving to: {output_path}")
    torch.save(ckpt, output_path)
    del ckpt
    gc.collect()

    out_size = os.path.getsize(output_path)
    in_size = os.path.getsize(input_path)
    print(f"Output size: {out_size / 1e9:.2f} GB "
          f"(reduced by {(1 - out_size / in_size) * 100:.0f}%)")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint.pth> [output.pth]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    strip_checkpoint(input_path, output_path)
