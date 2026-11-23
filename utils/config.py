import os


"""
bash
kaggle kernels push -p ./kaggle_launcher
kaggle kernels status "p00l000/kaggle-playground-11-26"
kaggle kernels output "p00l000/kaggle-playground-11-26" -p ./results
"""

import os
from utils.env import TABPFN_TOKEN 
ON_KAGGLE = os.path.exists("/kaggle/working")
DATASET_NAME = "kaggle-playground-11-26-perso/data"

if ON_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    EXPORT_DIR = "/kaggle/working"
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
    DATA_DIR = f"/kaggle/input/{DATASET_NAME}"
else:
    EXPORT_DIR = "/home/paul/pers/kaggle.playground.11.26/export"
    DATA_DIR = "/home/paul/pers/kaggle.playground.11.26/data"
    os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN
EXPORT_DIR = os.path.abspath(EXPORT_DIR)
os.makedirs(EXPORT_DIR, exist_ok=True)