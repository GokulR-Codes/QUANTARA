"""QAOA feature selection with PennyLane.

Problem (QUBO), x_i in {0,1} = "feature i is selected":

    minimise   C(x) = - sum_i r_i x_i  +  lambda * (sum_i x_i - k)^2

with r_i the relevance of feature i (computed on TRAINING data only) and k the number of
features to keep.  Substituting x_i = (1 - z_i) / 2 with z_i in {-1, +1} gives the Ising form

    C(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + c0

    h_i  = r_i / 2 + lambda * (k - n / 2)
    J_ij = lambda / 2
    c0   = - sum_i r_i / 2 + lambda * ((1 - 2k) n / 2 + n (n - 1) / 4 + k^2)

A computational-basis state |b> has z_i = 1 - 2 b_i, i.e. b_i = x_i.

The QAOA expectation <C> is computed **exactly** from the circuit's measurement-probability
distribution (no mean-field factorisation of <Z_i Z_j>).  The decoded selection is compared
with the exhaustive QUBO optimum and that comparison is reported — QAOA is a heuristic here
and nothing in this module claims a quantum advantage.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QAOAConfig:
    n_qubits: int = 4            # maximum number of candidate features (one qubit each)
    depth: int = 2
    backend: str = "default.qubit"
    analytic: bool = True
    shots: Optional[int] = None
    random_seed: int = 42
    sparsity_lambda: float = 1.0
    target_k: int = 2            # number of features to select
    max_iter: int = 200
    n_restarts: int = 1


@dataclass
class QAOAResult:
    selected_indices: list[int]
    selected_names: list[str]
    objective_value: float                 # optimised <C> (Ising expectation, incl. offset)
    optimization_duration_s: float
    n_qubits: int
    depth: int
    backend: str
    analytic: bool
    all_probabilities: list[float]
    gamma_params: list[float]
    beta_params: list[float]
    config: dict = field(default_factory=dict)
    # transparency / verification
    relevance: list[float] = field(default_factory=list)
    n_variables: int = 0
    decoded_probability: float = 0.0
    feasible_probability_mass: float = 0.0
    qubo_value_selected: float = 0.0
    qubo_value_optimal: float = 0.0
    matches_exhaustive_optimum: bool = False
    n_function_evaluations: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def qubo_objective(x: np.ndarray, relevance: np.ndarray, lam: float, k: int) -> float:
    return -float(relevance @ x) + lam * (float(x.sum()) - k) ** 2


def ising_coefficients(relevance: np.ndarray, lam: float, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (h, J, c0) such that  sum h_i z_i + sum_{i<j} J_ij z_i z_j + c0 == QUBO(x)."""
    r = np.asarray(relevance, dtype=float)
    n = len(r)
    h = r / 2.0 + lam * (k - n / 2.0)
    J = np.triu(np.full((n, n), lam / 2.0), k=1)
    c0 = -r.sum() / 2.0 + lam * ((1 - 2 * k) * n / 2.0 + n * (n - 1) / 4.0 + k ** 2)
    return h, J, float(c0)


def ising_energy_table(h: np.ndarray, J: np.ndarray, c0: float) -> np.ndarray:
    """Energy of every computational basis state (index bit 0 = most-significant = wire 0)."""
    n = len(h)
    energies = np.empty(2 ** n)
    for s in range(2 ** n):
        z = np.array([1 - 2 * ((s >> (n - 1 - i)) & 1) for i in range(n)], dtype=float)
        energies[s] = float(h @ z + z @ J @ z + c0)
    return energies


def exhaustive_qubo(relevance: np.ndarray, lam: float, k: int) -> tuple[np.ndarray, float]:
    n = len(relevance)
    best_x, best_v = np.zeros(n), float("inf")
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        v = qubo_objective(x, relevance, lam, k)
        if v < best_v - 1e-12:
            best_x, best_v = x, v
    return best_x, best_v


class QAOAFeatureSelector:
    """QAOA-based feature selection on a small candidate feature set."""

    def __init__(self, config: QAOAConfig) -> None:
        self.config = config
        # An explicit device seed stops PennyLane drawing from NumPy's *global* RNG
        # (its default behaviour), which would silently perturb the caller's random state.
        try:
            self._dev = qml.device(config.backend, wires=config.n_qubits, seed=config.random_seed)
        except TypeError:       # backend does not accept a seed argument
            self._dev = qml.device(config.backend, wires=config.n_qubits)
        self._h: Optional[np.ndarray] = None
        self._J: Optional[np.ndarray] = None
        self._c0: float = 0.0
        self._energies: Optional[np.ndarray] = None
        self._n_eval = 0

    # -- relevance -----------------------------------------------------------
    def compute_relevance_scores(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Between-class / total variance ratio per feature, min-max scaled to [0, 1].

        Call only with training-partition data.
        """
        n, d = features.shape
        overall_mean = features.mean(axis=0)
        overall_var = features.var(axis=0) + 1e-8
        between = np.zeros(d)
        for c in np.unique(labels):
            mask = labels == c
            between += mask.sum() * (features[mask].mean(axis=0) - overall_mean) ** 2
        between /= n
        f_scores = between / overall_var
        lo, hi = f_scores.min(), f_scores.max()
        return (f_scores - lo) / (hi - lo) if hi > lo else np.ones(d) / d

    # -- Hamiltonian ---------------------------------------------------------------
    def _build_qubo_coefficients(self, relevance_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ising (h, J) for the configured lambda / target_k (offset kept in ``_c0``)."""
        h, J, c0 = ising_coefficients(relevance_scores, self.config.sparsity_lambda, self.config.target_k)
        self._c0 = c0
        return h, J

    # -- circuit ----------------------------------------------------------------------
    def _probabilities(self, gamma: np.ndarray, beta: np.ndarray, h: np.ndarray, J: np.ndarray) -> np.ndarray:
        n = len(h)
        p = self.config.depth

        @qml.qnode(self._dev)
        def circuit():
            for i in range(n):
                qml.Hadamard(wires=i)
            for layer in range(p):
                for i in range(n):                        # cost layer: exp(-i gamma H_C)
                    qml.RZ(2.0 * gamma[layer] * h[i], wires=i)
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(J[i, j]) > 1e-12:
                            qml.CNOT(wires=[i, j])
                            qml.RZ(2.0 * gamma[layer] * J[i, j], wires=j)
                            qml.CNOT(wires=[i, j])
                for i in range(n):                        # mixer: exp(-i beta sum X)
                    qml.RX(2.0 * beta[layer], wires=i)
            return qml.probs(wires=list(range(n)))

        return np.asarray(circuit(), dtype=float)

    def _cost_hamiltonian_expectation(
        self, gamma: np.ndarray, beta: np.ndarray, h: np.ndarray, J: np.ndarray,
    ) -> float:
        """Exact <C> = sum_s P(s) C(s) from the circuit's probability distribution."""
        probs = self._probabilities(np.atleast_1d(gamma), np.atleast_1d(beta), h, J)
        energies = ising_energy_table(h, J, self._c0)
        self._n_eval += 1
        return float(probs @ energies)

    # -- driver ----------------------------------------------------------------------------
    def run(
        self,
        candidate_features: np.ndarray,
        candidate_labels: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> QAOAResult:
        n_cfg = self.config.n_qubits
        d = candidate_features.shape[1]
        if d > n_cfg:
            raise ValueError(
                f"Number of candidate features ({d}) exceeds qubit count ({n_cfg}). "
                "Reduce candidate features or increase qubits."
            )
        k = self.config.target_k
        if not (1 <= k <= d):
            raise ValueError(f"target_k must be between 1 and the number of features ({d}); got {k}")
        names = feature_names or [f"feature_{i}" for i in range(d)]
        if len(names) != d:
            raise ValueError(f"feature_names has {len(names)} entries for {d} features")

        relevance = self.compute_relevance_scores(candidate_features, candidate_labels)
        h, J = self._build_qubo_coefficients(relevance)
        self._h, self._J = h, J
        p = self.config.depth
        self._n_eval = 0

        def objective(params: np.ndarray) -> float:
            return self._cost_hamiltonian_expectation(params[:p], params[p:], h, J)

        rng = np.random.RandomState(self.config.random_seed)
        t0 = time.monotonic()
        best = None
        for _ in range(max(1, self.config.n_restarts)):
            x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
            res = minimize(objective, x0, method="COBYLA",
                           options={"maxiter": self.config.max_iter, "rhobeg": 0.5})
            if best is None or res.fun < best.fun:
                best = res
        elapsed = time.monotonic() - t0
        assert best is not None
        opt_gamma, opt_beta = best.x[:p], best.x[p:]
        probs = self._probabilities(opt_gamma, opt_beta, h, J)

        # Decode: most probable bitstring with exactly k ones
        best_state, best_prob, feasible_mass = None, -1.0, 0.0
        for s in range(2 ** d):
            bits = [(s >> (d - 1 - i)) & 1 for i in range(d)]
            if sum(bits) == k:
                feasible_mass += probs[s]
                if probs[s] > best_prob:
                    best_prob, best_state = probs[s], bits
        assert best_state is not None  # always exists because 1 <= k <= d
        selected = [i for i, b in enumerate(best_state) if b == 1]

        x_sel = np.zeros(d)
        x_sel[selected] = 1.0
        lam = self.config.sparsity_lambda
        v_sel = qubo_objective(x_sel, relevance, lam, k)
        x_opt, v_opt = exhaustive_qubo(relevance, lam, k)
        logger.info(
            "QAOA selected %s (QUBO %.4f; exhaustive optimum %.4f; P(decoded)=%.3f; %d evals)",
            [names[i] for i in selected], v_sel, v_opt, best_prob, self._n_eval,
        )
        return QAOAResult(
            selected_indices=selected,
            selected_names=[names[i] for i in selected],
            objective_value=float(best.fun),
            optimization_duration_s=elapsed,
            n_qubits=n_cfg,
            depth=p,
            backend=self.config.backend,
            analytic=self.config.analytic,
            all_probabilities=probs.tolist(),
            gamma_params=opt_gamma.tolist(),
            beta_params=opt_beta.tolist(),
            config=asdict(self.config),
            relevance=relevance.tolist(),
            n_variables=d,
            decoded_probability=float(best_prob),
            feasible_probability_mass=float(feasible_mass),
            qubo_value_selected=float(v_sel),
            qubo_value_optimal=float(v_opt),
            matches_exhaustive_optimum=bool(np.array_equal(x_sel, x_opt)),
            n_function_evaluations=self._n_eval,
        )
