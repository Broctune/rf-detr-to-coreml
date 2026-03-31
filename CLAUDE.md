# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Exports RF-DETR v1.5.1 detection and segmentation models to Apple CoreML format via runtime monkey-patching. The key insight: RF-DETR's deformable attention uses rank-6 tensors and bicubic interpolation, both unsupported by CoreML. Instead of forking upstream rfdetr, we apply monkey-patches at import time to make the model CoreML-compatible.

## Commands

```bash
# Install (editable)
pip install -e .

# Export a model (CLI)
rfdetr-coreml --model nano
rfdetr-coreml --model seg-nano
rfdetr-coreml --model all --output-dir output

# Or run directly
python export_coreml.py --model nano

# Run accuracy tests (requires macOS, exports all models and compares against PyTorch)
python scripts/test_export.py

# Latency benchmark
python scripts/benchmark_latency.py

# FP16 precision tests
python scripts/test_fp16.py
```

## Architecture

The package (`rfdetr_coreml/`) is a monkey-patch overlay on the upstream `rfdetr` library:

- **`__init__.py`** — Importing the package auto-applies all patches (both coremltools fixes and rfdetr patches). This is critical: `import rfdetr_coreml` must happen before any rfdetr model operations.
- **`patches.py`** — Three patches applied to upstream rfdetr:
  - **Patch A**: Replaces `MSDeformAttn.forward` to merge batch+heads dimensions, keeping all tensors at rank-5 or below (CoreML limit).
  - **Patch B**: Replacement core attention function (`_ms_deform_attn_core_5d`) that works with the merged 5D tensors. Called internally by Patch A.
  - **Patch C**: Replaces bicubic interpolation with bilinear in DinoV2 backbone (CoreML doesn't support bicubic).
- **`coreml_fixes.py`** — Fixes two coremltools bugs: `_cast` failing on shape-(1,) numpy arrays, and `view` failing on non-scalar shape Vars. Patches the coremltools op registry directly.
- **`export.py`** — `MODEL_REGISTRY` maps model names to classes and resolutions. `NormalizedWrapper` bakes ImageNet normalization into the model graph. `export_to_coreml()` handles: instantiate model → deepcopy → eval → export mode → wrap → trace → ct.convert → save.
- **`cli.py`** — argparse CLI entry point, registered as `rfdetr-coreml` console script.

## Critical Constraints

- **FP32 only**: FP16 causes catastrophic accuracy loss due to `F.grid_sample` precision sensitivity in deformable attention. Never default to FP16.
- **Patch order matters**: coremltools patches must be applied before rfdetr patches. Both are applied automatically by `import rfdetr_coreml`.
- **All patches use idempotent guards** (`_applied` flag) to prevent double-patching.
- **batch=1 uses `ct.ImageType`**, batch>1 uses `ct.TensorType` with float32 NCHW [0,1] input.
- **CoreML models should use `.all` or `.cpuAndGPU` compute units** — Neural Engine provides zero benefit because FP32 models can't run on ANE hardware.

## Dependencies

Requires macOS for CoreML conversion. Python >=3.10, torch >=2.4.0, coremltools >=8.0, rfdetr >=1.5.0. Tested with Python 3.12, torch 2.7.0, coremltools 8.1, rfdetr 1.5.1.
