"""Fit the final selectors on all full-mode tasks and save models/metapilot_models.joblib.

Usage:  python -m src.train_final [--mode full]
"""
from src.config import get_mode_from_args
from src.recommend import train_final

if __name__ == "__main__":
    print(f"saved {train_final(get_mode_from_args())}")
