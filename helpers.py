"""
Utility functions for RUL prediction models
"""

import os
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path

def seed_everything(seed=1234):
    """Set seeds for reproducibility across all libraries"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    """Seed worker processes for DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """Get available device (GPU or CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_dataset(csv_path):
    """Load RUL dataset from CSV"""
    return pd.read_csv(csv_path)


def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model, filepath):
    """Save model state dict"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    """Load model state dict"""
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")
    return model


def save_predictions(predictions, targets, filepath):
    """Save predictions and targets to CSV"""
    results_df = pd.DataFrame({
        'predictions': predictions,
        'targets': targets,
        'error': np.abs(predictions - targets)
    })
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def print_model_summary(model, model_name):
    """Print model summary"""
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(model)
    print(f"{'='*60}")
    print(f"Total Parameters: {n_params:,}")
    print(f"{'='*60}\n")

def verify_seed_model_structure(SEEDS=None, models_base_dir='Hybrid/models'):
    """Verify and list all seed-based model organization"""
    print("\n" + "="*80)
    print("SEED-BASED MODEL ORGANIZATION VERIFICATION")
    print("="*80)

    if not Path(models_base_dir).exists():
        print(f"\nModels directory not found: {models_base_dir}")
        return

    # Only show seed directories defined in SEEDS
    if SEEDS is None:
        print("\nNo SEEDS defined. Please set SEEDS to view seed directories.")
        return
    seed_set = {str(s) for s in SEEDS}
    seed_dirs = sorted([
        d for d in os.listdir(models_base_dir)
        if os.path.isdir(os.path.join(models_base_dir, d))
        and d.startswith('seed')
        and d.replace('seed', '') in seed_set
    ], key=lambda d: int(d.replace('seed', '')))

    print(f"\nFound {len(seed_dirs)} seed directories (from SEEDS):\n")

    for seed_dir in seed_dirs:
        seed_path = os.path.join(models_base_dir, seed_dir)
        seed_num = seed_dir.replace('seed', '')
        models_in_dir = [f for f in os.listdir(seed_path) if f.endswith('.pth')]

        # Check if seed matches any in SEEDS
        current_marker = " (IN SEEDS)" if seed_num in seed_set else ""
        print(f"{seed_dir:<20}{current_marker}")
        print(f"   Location: {seed_path}")
        print(f"   Models: {len(models_in_dir)}")

        if len(models_in_dir) > 0:
            print("   Files:")
            for model_file in sorted(models_in_dir)[:5]:  # Show first 5
                file_size = os.path.getsize(os.path.join(seed_path, model_file)) / 1024 / 1024  # MB
                print(f"      - {model_file} ({file_size:.2f} MB)")
            if len(models_in_dir) > 5:
                print(f"      ... and {len(models_in_dir) - 5} more models")
        else:
            print("   No models saved yet")
        print()


class EarlyStoppingCallback:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        elif val_loss > self.best_val_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0
        return self.early_stop
