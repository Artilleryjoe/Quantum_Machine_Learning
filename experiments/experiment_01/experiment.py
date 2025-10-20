"""Quantum kernel-based anomaly detection experiment.

This module demonstrates how quantum kernel estimation (QKE) can be used
as a precursor for parameterized quantum circuits (PQCs) in anomaly
identification tasks. The experiment trains a one-class support vector
machine (SVM) using kernel values generated from a fidelity-based quantum
kernel. The resulting decision function is evaluated on a mixture of
nominal and anomalous data points and compared to a classical baseline.

The goal is to highlight the relationship between feature map design in
quantum kernels and the ansätze typically used for PQCs, showing how QKE
provides intuition about circuit depth, entanglement layout, and feature
encodings prior to attempting a full variational optimization.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def generate_synthetic_dataset(
    *,
    n_nominal: int = 120,
    n_anomalous: int = 30,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a toy dataset for anomaly detection.

    Nominal data are sampled from a correlated Gaussian distribution, while
    anomalous samples come from a uniform distribution that spreads across
    the entire feature range. Labels are +1 for nominal points and -1 for
    anomalies to align with the `OneClassSVM` convention.
    """

    rng = np.random.default_rng(seed)
    mean = np.array([0.5, -0.2])
    cov = np.array([[0.2, 0.15], [0.15, 0.3]])
    nominal = rng.multivariate_normal(mean, cov, size=n_nominal)

    lower_bounds = mean - 2.5 * np.sqrt(np.diag(cov))
    upper_bounds = mean + 2.5 * np.sqrt(np.diag(cov))
    anomalous = rng.uniform(lower_bounds, upper_bounds, size=(n_anomalous, 2))

    data = np.vstack([nominal, anomalous])
    labels = np.concatenate([np.ones(n_nominal), -np.ones(n_anomalous)])
    is_nominal = np.array([True] * n_nominal + [False] * n_anomalous)
    return data, labels, is_nominal


@dataclasses.dataclass
class QuantumAnomalyDetectionResult:
    """Container for anomaly detection metrics."""

    train_fraction: float
    quantum_report: str
    classical_report: str


def build_quantum_kernel(feature_dimension: int, reps: int = 2) -> FidelityQuantumKernel:
    """Construct a fidelity-based quantum kernel instance.

    The feature map used here is a ZZFeatureMap, which mirrors the entangled
    encoding patterns that are often repurposed as the data-embedding layer in
    parameterized quantum circuits (PQCs). By experimenting with different
    repetitions (`reps`) or entanglement structures, practitioners can explore
    circuit expressivity prior to tuning trainable parameters in a PQC.
    """

    feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps)
    backend = AerSimulator()
    return FidelityQuantumKernel(feature_map=feature_map, backend=backend)


def evaluate_quantum_kernel_svm(
    kernel: FidelityQuantumKernel, train_data: np.ndarray, test_data: np.ndarray
) -> Tuple[OneClassSVM, np.ndarray]:
    """Train a one-class SVM with a precomputed quantum kernel."""

    k_train = kernel.evaluate(x_vec=train_data)
    svm = OneClassSVM(kernel="precomputed", nu=0.1, gamma="auto")
    svm.fit(k_train)

    k_test = kernel.evaluate(x_vec=test_data, y_vec=train_data)
    predictions = svm.predict(k_test)
    return svm, predictions


def evaluate_classical_baseline(
    train_data: np.ndarray, test_data: np.ndarray
) -> Tuple[OneClassSVM, np.ndarray]:
    """Train a classical RBF-kernel SVM for comparison."""

    svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    svm.fit(train_data)
    predictions = svm.predict(test_data)
    return svm, predictions


def run_experiment(
    *,
    train_fraction: float = 0.75,
    seed: int = 7,
) -> QuantumAnomalyDetectionResult:
    """Execute the anomaly detection workflow.

    Steps
    -----
    1. Generate synthetic data and split into train/test subsets.
    2. Scale features to the unit interval to mimic typical feature map bounds.
    3. Estimate a quantum kernel matrix using a ZZFeatureMap.
    4. Train a one-class SVM on nominal samples with the precomputed kernel.
    5. Evaluate the model on mixed nominal/anomalous points and report metrics.
    6. Compare against a classical RBF-kernel SVM baseline.
    """

    if not 0.1 <= train_fraction <= 0.95:
        raise ValueError("train_fraction should be between 0.1 and 0.95 for stability.")

    algorithm_globals.random_seed = seed
    raw_data, labels, is_nominal = generate_synthetic_dataset(seed=seed)

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    scaled_data = scaler.fit_transform(raw_data)

    nominal_data = scaled_data[is_nominal]
    n_train = int(len(nominal_data) * train_fraction)
    train_data = nominal_data[:n_train]
    test_data = scaled_data[n_train:]
    test_labels = labels[n_train:]

    quantum_kernel = build_quantum_kernel(feature_dimension=scaled_data.shape[1])
    _, quantum_predictions = evaluate_quantum_kernel_svm(quantum_kernel, train_data, test_data)

    _, classical_predictions = evaluate_classical_baseline(train_data, test_data)

    quantum_report = classification_report(
        test_labels, quantum_predictions, target_names=["Anomalous", "Nominal"], zero_division=0
    )
    classical_report = classification_report(
        test_labels, classical_predictions, target_names=["Anomalous", "Nominal"], zero_division=0
    )

    return QuantumAnomalyDetectionResult(
        train_fraction=train_fraction,
        quantum_report=quantum_report,
        classical_report=classical_report,
    )


def main() -> None:
    """Run the experiment and print results to stdout."""

    result = run_experiment()
    print("Quantum kernel anomaly detection report:\n")
    print(result.quantum_report)
    print("Classical RBF baseline report:\n")
    print(result.classical_report)


if __name__ == "__main__":
    main()
