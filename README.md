# SUTD 50.039 Deep Learning Project

This repository contains experiments and model training workflows for NASA turbofan Remaining Useful Life (RUL) prediction.

## 1) Environment Setup (Do This First)

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (cmd)
.\.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### CUDA Note

This project was developed with a CUDA-enabled setup. You must use a PyTorch/CUDA version that is compatible with your own GPU, CUDA driver, and OS.

If your machine does not support CUDA, use CPU-only PyTorch instead.

## 2) RNN Project Navigation (`RNN-based/`)

Use `RNN-based/` as the main area for the GRU/RNN/LSTM experiments.

- `archive/`: old exploratory files. You can ignore this folder.
- `best_model_and_evaluation/`: best GRU workflow.
  - `best-gru.ipynb`: trains the best GRU model using the selected optimal hyperparameters and produces training/loss curves.
  - `load_and_evaluate.ipynb`: loads the saved model from `best-gru.ipynb`, reproduces testing, and shows example predictions.
- `tuning/`: hyperparameter search and multi-dataset model training used to find the final comparison setup, together with loss curves.

## 3) Hybrid Transformer Project Navigation (`owen/`)

Use `owen/` for hybrid Transformer-based experiments (LSTM/GRU/CNN encoders with Transformer blocks).

- `archive/`: older files and experiments.
- `model_exploration.ipynb`: baseline end-to-end workflow (EDA, sequence creation, hybrid model definitions, training pipeline).
- `FD001_hyperparamter_tunning.ipynb`: FD001 hyperparameter tuning workflow (focus on sequence length and learning rate, with validation/test RMSE tracking).
- `hybrid_transformers.ipynb`: hybrid Transformer training workflow on processed NASA RUL data.
- `hybrid_transformers_FE.ipynb`: feature-engineered variant across multiple processed dataset versions.
- `FD001_hybrid_transformers_FE_AWS_SagemakerAI.ipynb`: FD001 feature-engineered hybrid Transformer workflow adapted for cloud/AWS-style execution.
- `FD002_hybrid_transformers_FE_AWS_SagemakerAI.ipynb`: FD002 counterpart of the AWS/feature-engineered workflow.
- `config.py` and `utils.py`: shared configuration and utility helpers for reproducible runs.

## 4) Transformer-Only Experiment (`Transformer/`)

- `best_transformer_hopt_lowvar_FINAL.pth`: saved best checkpoint weights from hyperparameter optimisation run.
- `hopt_lowvar_results.csv`: saved results from hyperparameter optimisation runs.
- `transformer-with-opt.ipynb`: standalone Transformer model workflow with hyperparameter optimization loop and evaluation logging, with plotted graph of best model's RMSE Loss.
- `recreate-transformer-inference.ipynb`: notebook that takes the best hyperparameters found in `transformer-with-opt.ipynb` and `best_transformer_hopt_lowvar_FINAL.pth` to reproduce metrics.
- `transformer_results_detailed.csv`: saved results (all seeds) from various dataset runs.
- `transformer_results_detailed.csv`: saved results (mean average) from various dataset runs.

Note on Report Metrics: The report cites a Best Test RMSE of 12.10 for RULTransformer. This reflects the peak test performance observed during the hyperparameter grid search (seed=1234, seq_len=50, lr=0.001, num_layers=1). The final reproducible model, selected strictly via best Validation RMSE to prevent data leakage, achieved a Test RMSE of 12.41.

## 5) Data Processing Notebooks (`data_preparation/`)

- `data_cleaning_1.ipynb`: generates cleaned RUL-labeled CSVs (linear and piecewise targets, normalized/non-normalized variants) from raw NASA files.
- `feature_engineering_2.ipynb`: feature engineering workflows (including filtering/selection variants) and exports processed datasets used by training notebooks.
- Processed files are stored under `data/processed-nasa-data/` (not in `data_preparation/output/`).

## 6) Suggested Reading Order

1. `data_preparation/data_cleaning_1.ipynb` and `data_preparation/feature_engineering_2.ipynb` (prepare and export datasets into `data/processed-nasa-data/`).
2. `RNN-based/tuning/` notebooks (RNN hyperparameter and dataset exploration).
3. `RNN-based/best_model_and_evaluation/best-gru.ipynb` and `RNN-based/best_model_and_evaluation/load_and_evaluate.ipynb` (final RNN training and evaluation).
4. `Transformer/transformer-with-opt.ipynb` (standalone Transformer optimization and evaluation).
5. `owen/model_exploration.ipynb`, then `owen/FD001_hyperparamter_tunning.ipynb` and FE/AWS notebooks (Hybrid Transformer workflows).
