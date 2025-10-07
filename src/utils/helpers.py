# src/utils/helpers.py
from pathlib import Path


def get_root_dir():
    return Path(__file__).resolve().parent.parent.parent  # src/utils -> root
