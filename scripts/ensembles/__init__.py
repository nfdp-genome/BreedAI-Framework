"""Ensemble methods for BreedAI."""

from .stacking import fit_nonneg_ridge, predict_stacker, summarize_weights

__all__ = ["fit_nonneg_ridge", "predict_stacker", "summarize_weights"]
