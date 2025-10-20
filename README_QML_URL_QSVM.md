# QML for Cybersecurity: URL Classification with QSVM

**Created:** 2025-10-20 15:25:46

This package contains:
- `QML_Cyber_URL_QSVM.ipynb` — a self-contained Jupyter notebook.
- `urls_sample.csv` — a synthetic sample dataset of 600 URL rows with labels.

## Prereqs (WSL Ubuntu)
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

## Create & activate a virtual environment
```bash
python3 -m venv qml_env
source qml_env/bin/activate
python -m pip install --upgrade pip
pip install qiskit qiskit-machine-learning scikit-learn pandas numpy matplotlib notebook
```

## Launch Jupyter and open the notebook
```bash
jupyter notebook
```
- Open `QML_Cyber_URL_QSVM.ipynb`
- If you want to use your own dataset, place a CSV with columns `url,label` next to the notebook and update `CSV_PATH` in the notebook.

## Run order
1. Read the **0) Environment Setup** cell (instructions), but you've already installed above.
2. Run the **Imports** cell.
3. Run **Load Data** (it will use `urls_sample.csv` by default).
4. Run **Feature Extraction** and inspect feature matrix shape.
5. Run **Train/Test Split and Scaling**.
6. Run **Classical Baseline (SVM)** — note accuracy/metrics.
7. Run **Quantum Kernel + QSVM** — start with `d=4` and `reps=2`.
8. Compare results and timing.

## Notes
- Simulator performance drops as you increase qubits (`d`) and circuit depth (`reps`). Keep it small.
- This is a didactic demo, not a production detector.
- For real data, prefer strong baselines and cross-validation; treat QML as exploratory until hardware/software mature.

Enjoy the quantum spelunking.
