from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
HISTORY_START = "2022-01-01"  # warm-up history for lag/rolling features
MODEL_START = "2023-03-01"  # first month retained for modelling
MODEL_END = "2026-09-01"  # last complete month in the supplied snapshot
TRAIN_END = "2025-06-01"
VAL_START = "2025-07-01"
VAL_END = "2025-12-01"
TEST_START = "2026-01-01"
TEST_END = "2026-09-01"
