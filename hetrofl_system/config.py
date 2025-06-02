import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Dataset configuration
MAIN_DATASET_PATH = r"C:\Users\VICTUS\Desktop\XX_XX\data\NF-ToN-IoT-v3-cleaned.csv"
TARGET_COLUMN = "Attack"
COLUMNS_TO_DROP = ["Label"]

# Local model configurations
LOCAL_MODELS = {
    "xgboost": {
        "path": BASE_DIR / "xgboost",
        "model_file": "xgboost_model.pkl",
        "scaler_file": "scaler.pkl",
        "encoder_file": "label_encoder.pkl",
        "dataset_file": "xgboost_dataset.csv",
        "type": "xgboost"
    },
    "random_forest": {
        "path": BASE_DIR / "Random_forest",
        "model_file": "random_forest_model.pkl",
        "scaler_file": "scaler.pkl",
        "encoder_file": "label_encoder.pkl",
        "dataset_file": "Randomforst_dataset.csv",
        "type": "sklearn"
    },
    "catboost": {
        "path": BASE_DIR / "catboost",
        "model_file": "ensemble_model.pkl",
        "scaler_file": "scaler.pkl",
        "encoder_file": "label_encoder.pkl",
        "dataset_file": "catboost_dataset.csv",
        "type": "catboost"
    }
}

# Global model configuration
GLOBAL_MODEL_CONFIG = {
    "hidden_layers": [512, 256, 128, 64],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "epochs": 100,
    "early_stopping_patience": 10,
    "validation_split": 0.2
}

# Federated learning configuration
FL_CONFIG = {
    "num_rounds": 50,
    "aggregation_method": "fedavg",  # fedavg, weighted_avg
    "min_clients": 2,
    "client_fraction": 1.0,
    "local_epochs": 5,
    "knowledge_distillation_alpha": 0.7,
    "temperature": 3.0
}

# Performance tracking
METRICS_TO_TRACK = [
    "accuracy", "f1_score", "precision", "recall", 
    "roc_auc", "loss", "training_time"
]

# GUI configuration
GUI_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": True,
    "update_interval": 5000,  # milliseconds
    "max_plot_points": 100
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "hetrofl_system.log"
}

# Create directories if they don't exist
for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 