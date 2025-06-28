import os

# Base directory of the project (assumes this file lives in `config/`)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths used throughout the project
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# Optional: ensure folders exist
os.makedirs(RESULTS_DIR, exist_ok=True)
