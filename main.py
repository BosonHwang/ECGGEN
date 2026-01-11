"""Thin entrypoint to run training/eval from the repository root or inside ecggen/."""
import os
import sys

# Allow running as `python main.py` from within the ecggen/ directory by adding the parent to sys.path.
ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from ecggen.src.utils.trainer import main as trainer_main


if __name__ == "__main__":
    trainer_main()


