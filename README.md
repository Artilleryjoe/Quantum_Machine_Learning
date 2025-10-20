# Quantum Machine Learning Experiments

This repository collects small, self-contained demonstrations of quantum machine
learning (QML) techniques that target realistic cybersecurity analytics
scenarios. The latest addition focuses on **anomaly detection** and shows how
**quantum kernel estimation (QKE)** can act as a bridge toward more expressive
**parameterized quantum circuits (PQCs)**.

## Quantum kernel anomaly detection experiment

The experiment located at
[`experiments/quantum_kernel_anomaly_detection/experiment.py`](experiments/quantum_kernel_anomaly_detection/experiment.py)
implements a one-class support vector machine (SVM) that separates nominal
network behavior from anomalies. Instead of relying on a classical kernel, the
model uses the fidelity between quantum states prepared by a ZZFeatureMap to
measure similarity between samples.

Why start with a kernel? QKE offers a simulation-friendly way to explore circuit
expressivity, entanglement structure, and data-encoding strategies before
introducing trainable parameters. The same feature map that defines the kernel
can later be embedded into a PQC ansatz; kernel performance therefore provides
an evidence-based prior on whether the chosen layout is likely to separate the
data manifold.

### Workflow summary

1. **Synthetic dataset generation** – Multivariate Gaussian samples represent
   nominal traffic, while uniformly distributed points mimic anomalous activity.
2. **Feature scaling** – The data are normalized to match the input domain of
   common qubit encodings.
3. **Quantum kernel estimation** – A `ZZFeatureMap` with configurable depth and
   entanglement pattern prepares states whose pairwise fidelities populate the
   SVM's kernel matrix.
4. **One-class SVM training** – The SVM learns the boundary of normal behavior
   using only nominal samples and flags low-fidelity points as anomalies.
5. **Classical baseline comparison** – An RBF-kernel SVM runs on the same data to
   highlight where quantum-inspired encodings might yield different decision
   boundaries.

### Connecting QKE to PQCs

Feature maps in QKE are unparameterized circuits that encode classical data into
quantum states. When building a PQC for variational algorithms (e.g., quantum
neural networks), these feature maps often become the "data layer" that precedes
trainable blocks. Running the kernel experiment allows you to:

- **Stress-test encodings** – Observe how feature repetitions (`reps`) or
  entanglement topologies affect separation before adding variational layers.
- **Gauge resource demands** – Estimate qubit counts and depth requirements under
  noise-free simulation to plan for execution on noisy devices.
- **Bootstrap initializations** – Favor feature maps that already capture class
  structure, leading to easier optimization when parameters are introduced.

### Running the experiment

> **Note:** The script depends on `qiskit` and `qiskit-machine-learning` in
> addition to `scikit-learn` and `numpy`.

```bash
pip install qiskit qiskit-machine-learning scikit-learn numpy
python experiments/quantum_kernel_anomaly_detection/experiment.py
```

The script prints classification reports for both the quantum kernel model and
its classical RBF counterpart, allowing you to inspect precision/recall tradeoffs
for each class.

### Extending toward PQCs

After validating that the chosen feature map produces meaningful kernel
separations, you can refactor the circuit into a full PQC by appending a
parameterized ansatz (e.g., `TwoLocal`, HEA) and optimizing with gradient-based
or gradient-free routines. The kernel experiment therefore acts as a diagnostic
stage: if QKE fails to expose structure in the data, a PQC built on the same
encoding is unlikely to perform well without significant adjustments.
