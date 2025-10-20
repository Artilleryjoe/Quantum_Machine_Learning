"""Quantum kernel-based anomaly detection experiment (universal Qiskit version)."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Tuple

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

# --- Qiskit imports ---
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

try:
    # Newer placement (≥1.0)
    from qiskit_algorithms.utils import algorithm_globals
except ImportError:  # pragma: no cover - compatibility fallback
    from qiskit.utils import algorithm_globals


# ----------------------------
# Data generation
# ----------------------------
def generate_synthetic_dataset(
    *, n_nominal: int = 120, n_anomalous: int = 30, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ----------------------------
# Result container
# ----------------------------
@dataclasses.dataclass
class QuantumAnomalyDetectionResult:
    train_fraction: float
    quantum_report: str
    classical_report: str


# ----------------------------
# Quantum kernel builder
# ----------------------------
def build_quantum_kernel(feature_dimension: int, reps: int = 2) -> FidelityQuantumKernel:
    """Build a FidelityQuantumKernel that supports both backend and quantum_instance APIs."""

    feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps)
    backend = AerSimulator()

    params = inspect.signature(FidelityQuantumKernel.__init__).parameters

    # Newer QML uses `backend`
    if "backend" in params:
        return FidelityQuantumKernel(feature_map=feature_map, backend=backend)

    # Older QML uses `quantum_instance`
    if "quantum_instance" in params:

        class FakeQuantumInstance:
            """Shim to satisfy legacy FidelityQuantumKernel API expectations."""

            def __init__(self, backend):
                self._backend = backend

            @property
            def backend(self):
                return self._backend

        return FidelityQuantumKernel(feature_map=feature_map, quantum_instance=FakeQuantumInstance(backend))

    # Absolute fallback (rare edge case)
    return FidelityQuantumKernel(feature_map=feature_map)


# ----------------------------
# Evaluators
# ----------------------------
def evaluate_quantum_kernel_svm(
    kernel: FidelityQuantumKernel, train_data: np.ndarray, test_data: np.ndarray
) -> Tuple[OneClassSVM, np.ndarray]:
    k_train = kernel.evaluate(x_vec=train_data)
    svm = OneClassSVM(kernel="precomputed", nu=0.1, gamma="auto")
    svm.fit(k_train)

    k_test = kernel.evaluate(x_vec=test_data, y_vec=train_data)
    predictions = svm.predict(k_test)
    return svm, predictions


def evaluate_classical_baseline(
    train_data: np.ndarray, test_data: np.ndarray
) -> Tuple[OneClassSVM, np.ndarray]:
    svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    svm.fit(train_data)
    predictions = svm.predict(test_data)
    return svm, predictions


# ----------------------------
# Experiment runner
# ----------------------------
def run_experiment(
    *, train_fraction: float = 0.75, seed: int = 7
) -> QuantumAnomalyDetectionResult:
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


# ----------------------------
# Entry point
# ----------------------------
def main() -> None:
    result = run_experiment()
    print("Quantum kernel anomaly detection report:\n")
    print(result.quantum_report)
    print("Classical RBF baseline report:\n")
    print(result.classical_report)


if __name__ == "__main__":
    main()
