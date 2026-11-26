"""Shared model registry for the web demo.

Importing this module points rfdetr's weight cache (``RF_HOME``) at
``webapp/models`` so weights live inside the repo folder. It must be imported
before ``rfdetr``.
"""

import os
import threading
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
os.environ.setdefault("RF_HOME", str(MODELS_DIR))

APACHE = "Apache-2.0"
# Platform Model License 1.0: these come from the rfdetr_plus package (`rfdetr[plus]`).
# No API key is needed to download or run them, but the license requires a Roboflow
# platform plan (free or paid) in good standing.
PML = "PML-1.0"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    cls: str  # class name exported by rfdetr
    resolution: int
    weights: str  # filename inside RF_HOME
    license: str


# Smallest to largest.
MODELS = {
    "atto": ModelSpec("Atto", "RFDETRAtto", 380, "rf-detr-atto.pth", PML),
    "femto": ModelSpec("Femto", "RFDETRFemto", 384, "rf-detr-femto.pth", PML),
    "pico": ModelSpec("Pico", "RFDETRPico", 560, "rf-detr-pico.pth", PML),
    "nano": ModelSpec("Nano", "RFDETRNano", 384, "rf-detr-nano.pth", APACHE),
    "small": ModelSpec("Small", "RFDETRSmall", 512, "rf-detr-small.pth", APACHE),
    "medium": ModelSpec("Medium", "RFDETRMedium", 576, "rf-detr-medium.pth", APACHE),
    "xlarge": ModelSpec("XL", "RFDETRXLarge", 700, "rf-detr-xlarge.pth", PML),
    "2xlarge": ModelSpec("2XL", "RFDETR2XLarge", 880, "rf-detr-xxlarge.pth", PML),
}
DEFAULT_MODEL = "nano"

_instances = {}
_load_lock = threading.Lock()
_infer_locks = {name: threading.Lock() for name in MODELS}


def weights_dir() -> Path:
    """Directory rfdetr reads/writes weights in (honours a pre-set RF_HOME)."""
    return Path(os.environ["RF_HOME"]).expanduser()


def weights_path(name: str) -> Path:
    return weights_dir() / MODELS[name].weights


def get_model(name: str):
    """Return the cached model for ``name``, instantiating it on first use."""
    if name not in MODELS:
        raise KeyError(name)
    with _load_lock:
        if name not in _instances:
            import rfdetr

            # PML classes resolve through rfdetr's lazy export; ImportError if rfdetr_plus is missing.
            _instances[name] = getattr(rfdetr, MODELS[name].cls)()
        return _instances[name]


def infer_lock(name: str) -> threading.Lock:
    """Per-model lock: a model instance is not safe to run concurrently."""
    return _infer_locks[name]
