# Quantum Machine Learning Experiments

This repository collects small, self-contained demonstrations of quantum
machine learning (QML) techniques that target realistic cybersecurity
analytics scenarios.

## Repository structure

```
.
├── README.md
├── data/
│   └── urls_sample.csv
└── experiments/
    └── experiment_01/
        ├── README.md
        └── QML_Cyber_URL_QSVM.ipynb
```

- **data/** – Reusable datasets that support the experiments. The provided
  `urls_sample.csv` file contains a balanced selection of benign and malicious
  URLs that can be repurposed for feature-engineering or anomaly-detection
  exercises.
- **experiments/** – Each numbered folder captures an isolated study. The first
  experiment explores how quantum kernel methods can separate malicious web
  traffic from benign activity.

## First experiment: Quantum kernel anomaly detection

`experiments/experiment_01` contains a Jupyter notebook that walks through an
anomaly detection workflow based on quantum kernel estimation. The associated
README in that directory explains the problem motivation, notebook layout, and
required dependencies.

### Running the notebook

1. Create and activate a Python environment with Jupyter, Qiskit, and
   scikit-learn installed.
2. Launch Jupyter Lab or Jupyter Notebook from the repository root:

   ```bash
   jupyter lab
   ```

3. Open `experiments/experiment_01/QML_Cyber_URL_QSVM.ipynb` and run the cells
   sequentially.

The notebook demonstrates how quantum-enhanced similarity measures can augment
classical anomaly detectors and provides a baseline for future PQC-based
experiments.
