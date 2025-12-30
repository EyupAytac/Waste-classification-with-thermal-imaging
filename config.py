import torch
import os

# =====================================================
# PATHS & SYSTEM
# =====================================================
# Note: If running locally, remove the Colab Drive path
ROOT_DIR = "/content/drive/MyDrive/proje"
CACHE_DIR = "/content/drive/MyDrive/cache"
EXCEL_NAME = "datas.xlsx"
MODEL_OUT_NAME = "best_model_advanced_thermal.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# =====================================================
# DATA LABELS
# =====================================================
# PLEASE UPDATE THESE BASED ON YOUR EXCEL DATA
LABEL_MAP = {
    "Metal": 0,
    "Plastic": 1,
    "Cardboard": 2
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# =====================================================
# VIDEO PROTOCOL
# =====================================================
SKIP_SEC = 5.0
HEATING_SEC = 10.0
COOLING_SEC = 10.0

# =====================================================
# SAMPLING / PREPROCESSING
# =====================================================
MAX_FRAMES = 100
OUT_SIZE = 224

# =====================================================
# TRAINING SETTINGS
# =====================================================
TRAIN_SPLIT = 0.7
BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4

# =====================================================
# CACHE & DEBUG
# =====================================================
USE_CACHE = True
FORCE_REBUILD_CACHE = False
DATASET_PROGRESS_PRINT = False
CACHE_FORMAT = "pt"  # or 'npy'