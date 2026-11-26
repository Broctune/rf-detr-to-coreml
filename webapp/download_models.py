"""Pre-download RF-DETR weights into webapp/models.

Usage:
    webapp/.venv/bin/python webapp/download_models.py              # all models
    webapp/.venv/bin/python webapp/download_models.py atto nano    # only these
"""

import shutil
import sys
from pathlib import Path

from models import MODELS, PML, weights_dir, weights_path  # sets RF_HOME; import first

from rfdetr.assets.model_weights import download_pretrain_weights

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    names = sys.argv[1:] or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"Unknown model(s): {', '.join(unknown)}. Choose from: {', '.join(MODELS)}.")

    weights_dir().mkdir(parents=True, exist_ok=True)

    for name in names:
        dest = weights_path(name)
        # Reuse a copy already sitting in the repo root (rfdetr MD5-checks it below).
        seed = REPO_ROOT / dest.name
        if not dest.exists() and seed.is_file():
            print(f"{name}: copying {seed} -> {dest}")
            shutil.copy2(seed, dest)
        download_pretrain_weights(str(dest))

    print()
    for name in names:
        dest = weights_path(name)
        size = f"{dest.stat().st_size / 1e6:6.1f} MB" if dest.is_file() else "MISSING"
        print(f"{name:8s} {MODELS[name].license:10s} {size}  {dest}")
    if any(MODELS[n].license == PML for n in names):
        print("\nPML-1.0 models need a Roboflow platform plan in good standing (no API key).")


if __name__ == "__main__":
    main()
