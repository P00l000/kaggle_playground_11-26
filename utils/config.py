import os

ON_KAGGLE = os.path.exists("/kaggle/working")
DATASET_NAME = "kaggle_playground_11-26_hot_ones"

if ON_KAGGLE:
    EXPORT_DIR = "/kaggle/working"
    DATASET_NAME = "your-dataset-name"
    DATA_DIR = f"/kaggle/input/{DATASET_NAME}"
else:
    EXPORT_DIR = "/home/paul/pers/kaggle.playground.11.26/export"
    DATA_DIR = "/home/paul/pers/kaggle.playground.11.26/data"

EXPORT_DIR = os.path.abspath(EXPORT_DIR)
os.makedirs(EXPORT_DIR, exist_ok=True)