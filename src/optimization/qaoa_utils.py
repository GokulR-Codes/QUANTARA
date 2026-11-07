"""Utilities for QAOA objective construction and verification."""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np


def compute_qubo_objective(
    x: np.ndarray,
    relevance: np.ndarray,
    lam: float,
    k: int,
) -> float:
    """Compute the QUBO objective value for a binary vector x.
    Objective: -r^T x + lambda * (sum(x) - k)^2
    Lower is better.
    """
    return -float(relevance @ x) + lam * (float(x.sum()) - k) ** 2


def exhaustive_qubo_search(
    relevance: np.ndarray,
    lam: float,
    k: int,
) -> tuple[np.ndarray, float]:
    """Find the exact optimal QUBO solution by exhaustive enumeration.
    Only feasible for small n (<=20).
    Returns: (best_x, best_objective_value)
    """
    n = len(relevance)
    best_val = float("inf")
    best_x = np.zeros(n)
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        val = compute_qubo_objective(x, relevance, lam, k)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    return best_x, best_val


def verify_qaoa_against_exhaustive(
    relevance: np.ndarray,
    qaoa_selected: list[int],
    lam: float,
    k: int,
) -> dict:
    """Compare QAOA result against exhaustive search for small problems."""
    n = len(relevance)
    best_x, best_val = exhaustive_qubo_search(relevance, lam, k)

    qaoa_x = np.zeros(n)
    for i in qaoa_selected:
        if i < n:
            qaoa_x[i] = 1.0
    qaoa_val = compute_qubo_objective(qaoa_x, relevance, lam, k)

    return {
        "exhaustive_optimal_value": best_val,
        "exhaustive_selected": [i for i, b in enumerate(best_x) if b == 1],
        "qaoa_value": qaoa_val,
        "qaoa_selected": qaoa_selected,
        "approximation_ratio": qaoa_val / best_val if best_val != 0 else 1.0,
        "is_optimal": np.allclose(qaoa_x, best_x),
    }
