"""Paths and global constants. Override locations with BER_DATA_DIR / BER_WORK_DIR / BER_OUTPUT_DIR."""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]          # code/business_entity_resolution
REPO_DIR = CODE_DIR.parents[1]

DATA_DIR = Path(os.environ.get(
    "BER_DATA_DIR",
    REPO_DIR / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", CODE_DIR / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", REPO_DIR / "output"))

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)

SEED = 26


def raw_path(split, src):
    return DATA_DIR / split / f"{split}_source{src}.tsv"


def gt_raw_path():
    return DATA_DIR / "train" / "train_ground_truth.tsv"


def records_path(split, src):
    return WORK_DIR / "parquet" / f"{split}_s{src}.parquet"


def gt_pairs_path():
    return WORK_DIR / "parquet" / "train_gt_pairs.parquet"


def gt_entities_path():
    return WORK_DIR / "parquet" / "train_gt_entities.parquet"


def norm_path(split, src):
    return WORK_DIR / "norm" / f"{split}_s{src}.parquet"


def run_dir(run):
    # BER_RUN_SUFFIX keeps experiment runs apart (e.g. fitall_k2)
    suffix = os.environ.get("BER_RUN_SUFFIX", "")
    return WORK_DIR / "runs" / (f"{run}_{suffix}" if suffix else run)


# worker threads/processes; defaults to every core (override with BER_JOBS)
N_JOBS = int(os.environ.get("BER_JOBS", os.cpu_count() or 4))


def tag_suffix():
    """BER_TAG separates artefact sets (e.g. 'all' = final models on every training pair)."""
    tag = os.environ.get("BER_TAG", "")
    return f"_{tag}" if tag else ""
