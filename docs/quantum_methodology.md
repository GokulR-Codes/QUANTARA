# Quantum Methodology

## Overview

This project uses two quantum algorithms implemented with [PennyLane](https://pennylane.ai/) running on the `lightning.qubit` CPU simulator:

1. **QAOA** — Quantum Approximate Optimization Algorithm for feature selection
2. **VQC** — Variational Quantum Classifier for risk classification

Neither algorithm requires quantum hardware. All circuits are simulated classically.

---

## QAOA Feature Selection

### Problem Formulation

After feature fusion, the model produces a 64-dimensional fused vector. Not all 64 features are equally informative. We formulate feature selection as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem:

$$\min_{x \in \{0,1\}^n} \; x^T Q x$$

where $Q_{ij}$ encodes pairwise feature relevance (diagonal: individual relevance; off-diagonal: redundancy penalty).

### QUBO Construction

From a batch of fused feature vectors $\{f_1, \ldots, f_B\}$ and class labels:

1. **Individual relevance** $r_i$: Pearson correlation between feature $i$ and the label
2. **Redundancy** $\rho_{ij}$: Pearson correlation between features $i$ and $j$
3. **QUBO matrix**: $Q_{ii} = -r_i$,  $Q_{ij} = +\lambda |\rho_{ij}|$ for $i \neq j$

where $\lambda$ is a redundancy penalty weight.

### Ising Hamiltonian

The QUBO is mapped to an Ising Hamiltonian over $n$ qubits. For variable $x_i = (1 - Z_i)/2$:

$$H = \sum_i h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j$$

with coefficients derived from $Q$.

### QAOA Circuit

The QAOA circuit with depth $p$ alternates:
- **Cost layer**: $e^{-i \gamma H_C}$ — encodes the QUBO objective
- **Mixer layer**: $e^{-i \beta \sum_i X_i}$ — explores the solution space

Parameters $(\gamma_1, \ldots, \gamma_p, \beta_1, \ldots, \beta_p)$ are optimized classically using gradient-based optimizers.

After optimization, the most probable bitstring corresponds to the selected feature subset.

### Implementation

- **File**: `src/models/quantum/qaoa.py`
- **Backend**: `pennylane.device("lightning.qubit", wires=n)`
- **Layers**: Configurable (default: 3)
- **Optimizer**: Adam via PennyLane's optimization module
- **Output**: Binary selection mask of length $n$; report includes gap to exhaustive QUBO optimum

---

## VQC Classification

### Circuit Design

The Variational Quantum Classifier encodes selected features as qubit rotations and applies a parameterized ansatz:

1. **Data encoding**: Angle embedding — feature $x_i$ → $R_Y(x_i)$ on qubit $i$
2. **Variational layer** (repeated $L$ times):
   - $R_Y(\theta_{i,l})$ on each qubit
   - CNOT ladder: qubit $i$ → qubit $(i+1) \mod n$
3. **Measurement**: Pauli-Z expectation value $\langle Z_0 \rangle$ → binary output

### Training

The VQC is trained using the **parameter-shift rule** (exact quantum gradient):

$$\frac{\partial \langle H \rangle}{\partial \theta} = \frac{1}{2}\left[\langle H \rangle_{\theta + \pi/2} - \langle H \rangle_{\theta - \pi/2}\right]$$

Loss: Binary cross-entropy between $\langle Z_0 \rangle$ and the true label.

### Dtype Notes

PennyLane's `lightning.qubit` simulator returns `float64` measurements. PyTorch linear layers require `float32`. The VQC forward pass explicitly casts: `.float()` after stacking measurements.

### Implementation

- **File**: `src/models/quantum/vqc.py`
- **Backend**: `pennylane.device("lightning.qubit", wires=n_qubits)`
- **Qubits**: Equals number of QAOA-selected features
- **Layers**: Configurable (default: 3)
- **Optimizer**: Adam (classical), parameter-shift gradients

---

## Federated Learning Integration

In the federated setting, the CNN, LSTM, and fusion modules are trained across simulated clients using **Federated Averaging (FedAvg)**:

1. Server broadcasts current global weights
2. Each client trains on its local (synthetic) data for $E$ local epochs
3. Clients send weight updates to server
4. Server aggregates: $w_{\text{global}} = \sum_k \frac{n_k}{N} w_k$

After federated training, the global fused representation is passed to QAOA for feature selection and then VQC training.

### Implementation

- **File**: `src/federated/fedavg.py`
- **Clients**: Simulated locally (no real network)
- **Data**: Synthetic; split randomly across clients

---

## Limitations and Honest Claims

| Claim | Status |
|---|---|
| Circuit is correct (Ising coefficients, sign convention) | ✅ Verified + tested vs exhaustive QUBO solver |
| PennyLane runs without error | ✅ Verified on CPU simulator |
| QAOA finds optimal subset | ⚠️ Heuristic — approximation quality depends on circuit depth and optimization iterations |
| VQC generalizes to real data | ❌ Not tested — no labelled dataset available |
| Quantum advantage over classical baseline | ❌ Not claimed — this is a proof-of-concept research pipeline |
| Accuracy/AUC figures are real | ❌ All metrics are on synthetic random data — not clinically meaningful |

> **All experiment results are clearly labelled `synthetic: true` in the API response.**
