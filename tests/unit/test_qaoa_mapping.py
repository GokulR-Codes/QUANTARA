"""QAOA correctness: Ising mapping, circuit expectation, decoding, side effects."""
import itertools

import numpy as np
import pytest

from src.models.quantum.qaoa import (
    QAOAConfig, QAOAFeatureSelector, exhaustive_qubo, ising_coefficients,
    ising_energy_table, qubo_objective,
)


@pytest.mark.parametrize("n,k,lam", [(3, 1, 1.0), (4, 2, 1.0), (5, 2, 0.7), (5, 3, 2.5), (6, 3, 1.3)])
def test_ising_energy_equals_qubo_for_every_bitstring(n, k, lam):
    r = np.random.RandomState(n * 10 + k).rand(n)
    E = ising_energy_table(*ising_coefficients(r, lam, k))
    for s, bits in enumerate(itertools.product([0, 1], repeat=n)):
        assert E[s] == pytest.approx(qubo_objective(np.array(bits, float), r, lam, k), abs=1e-9)


def test_ground_state_is_the_qubo_optimum_not_its_complement():
    r = np.array([0.9, 0.7, 0.2, 0.05])
    E = ising_energy_table(*ising_coefficients(r, 1.0, 2))
    best = int(E.argmin())
    bits = tuple((best >> (3 - i)) & 1 for i in range(4))
    assert bits == (1, 1, 0, 0)                         # the two MOST relevant features
    x_opt, _ = exhaustive_qubo(r, 1.0, 2)
    assert tuple(int(v) for v in x_opt) == bits


def _numpy_qaoa_probs(h, J, gamma, beta):
    """Independent state-vector QAOA (no PennyLane): |+>^n, cost phases, X-mixers."""
    n, p = len(h), len(gamma)
    dim = 2 ** n
    zs = np.array([[1 - 2 * ((s >> (n - 1 - i)) & 1) for i in range(n)] for s in range(dim)], float)
    e_noconst = zs @ h + np.einsum("si,ij,sj->s", zs, J, zs)
    psi = np.full(dim, 1 / np.sqrt(dim), complex)
    c, sn = np.cos, np.sin
    for layer in range(p):
        psi = psi * np.exp(-1j * gamma[layer] * e_noconst)
        rx = np.array([[c(beta[layer]), -1j * sn(beta[layer])], [-1j * sn(beta[layer]), c(beta[layer])]])
        psi = psi.reshape([2] * n)
        for q in range(n):
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [q])), 0, q)
        psi = psi.reshape(dim)
    return np.abs(psi) ** 2


@pytest.mark.parametrize("p", [1, 2])
def test_circuit_expectation_matches_independent_numpy_simulation(p):
    n, k = 4, 2
    r = np.array([0.9, 0.6, 0.3, 0.1])
    sel = QAOAFeatureSelector(QAOAConfig(n_qubits=n, depth=p, target_k=k))
    h, J = sel._build_qubo_coefficients(r)
    rng = np.random.RandomState(7)
    gamma, beta = rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)
    probs_pl = sel._probabilities(gamma, beta, h, J)
    probs_np = _numpy_qaoa_probs(h, J, gamma, beta)
    assert np.allclose(probs_pl, probs_np, atol=1e-8)
    expect_np = float(probs_np @ ising_energy_table(h, J, sel._c0))
    assert sel._cost_hamiltonian_expectation(gamma, beta, h, J) == pytest.approx(expect_np, abs=1e-8)


def _separable_problem(rng, n=4, per_class=30):
    """Feature 0,1 informative; 2,3 noise."""
    y = np.repeat([0, 1, 2], per_class)
    X = rng.normal(0, 1, (len(y), n))
    X[:, 0] += 2.5 * y
    X[:, 1] += 2.0 * y
    return X, y


def test_run_returns_feasible_selection_and_honest_comparison_to_exhaustive():
    X, y = _separable_problem(np.random.RandomState(0))
    res = QAOAFeatureSelector(QAOAConfig(n_qubits=4, depth=2, target_k=2, max_iter=120)).run(X, y, list("abcd"))
    assert len(res.selected_indices) == 2 and res.selected_names == [list("abcd")[i] for i in res.selected_indices]
    assert res.qubo_value_optimal <= res.qubo_value_selected + 1e-12      # optimum is a lower bound
    assert 0 < res.decoded_probability <= res.feasible_probability_mass <= 1 + 1e-9
    assert sum(res.all_probabilities) == pytest.approx(1.0)
    assert res.matches_exhaustive_optimum == (abs(res.qubo_value_selected - res.qubo_value_optimal) < 1e-9)
    feasible = [qubo_objective(np.array(b, float), np.array(res.relevance), 1.0, 2)
                for b in itertools.product([0, 1], repeat=4) if sum(b) == 2]
    assert res.qubo_value_selected <= np.mean(feasible)                   # better than a random feasible pick
    assert res.n_function_evaluations > 5


def test_selection_is_deterministic_for_a_seed():
    X, y = _separable_problem(np.random.RandomState(1))
    cfg = dict(n_qubits=4, depth=1, target_k=2, max_iter=40, random_seed=3)
    a = QAOAFeatureSelector(QAOAConfig(**cfg)).run(X, y)
    b = QAOAFeatureSelector(QAOAConfig(**cfg)).run(X, y)
    assert a.selected_indices == b.selected_indices and a.objective_value == pytest.approx(b.objective_value)


def test_does_not_reseed_numpy_global_rng():
    X, y = _separable_problem(np.random.RandomState(2))
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    QAOAFeatureSelector(QAOAConfig(n_qubits=4, depth=1, target_k=2, max_iter=10)).run(X, y)
    assert np.random.rand() == expected


def test_fewer_features_than_qubits_uses_only_real_features():
    X, y = _separable_problem(np.random.RandomState(3), n=3)
    res = QAOAFeatureSelector(QAOAConfig(n_qubits=4, depth=1, target_k=2, max_iter=30)).run(X, y)
    assert res.n_variables == 3 and len(res.selected_indices) == 2 and max(res.selected_indices) < 3
    assert len(res.all_probabilities) == 2 ** 3


@pytest.mark.parametrize("n_feat,k,names,msg", [(5, 2, None, "exceeds qubit count"),
                                                (3, 4, None, "target_k"), (3, 0, None, "target_k"),
                                                (3, 2, ["a", "b"], "feature_names")])
def test_invalid_inputs_raise(n_feat, k, names, msg):
    X = np.random.RandomState(0).rand(10, n_feat)
    y = np.arange(10) % 2
    with pytest.raises(ValueError, match=msg):
        QAOAFeatureSelector(QAOAConfig(n_qubits=4, depth=1, target_k=k)).run(X, y, names)
