"""Dataset utilities for the age estimation dissertation project.

This module creates age groups, parses dataset metadata, and prepares CSV files
for Keras/TensorFlow data generators.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd


AGE_GROUPS = [
    (0, 2, "0_2"),
    (3, 5, "3_5"),
    (6, 13, "6_13"),
    (14, 18, "14_18"),
    (19, 24, "19_24"),
    (25, 33, "25_33"),
    (34, 48, "34_48"),
    (49, 64, "49_64"),
    (65, 200, "65_plus"),
]


def age_to_group(age: int | float) -> int:
    """Convert an exact age into one of 9 age-group IDs."""
    age = int(age)
    if age < 0:
        raise ValueError(f"Age cannot be negative: {age}")
    for group_id, (low, high, _) in enumerate(AGE_GROUPS):
        if low <= age <= high:
            return group_id
    raise ValueError(f"Age is outside supported range: {age}")


def combined_label(age_group: int, gender: int) -> int:
    """Combine 9 age groups and 2 genders into 18 classes.

    Gender convention used in the dissertation:
    0 = female, 1 = male.
    """
    age_group = int(age_group)
    gender = int(gender)
    if age_group not in range(9):
        raise ValueError(f"Invalid age_group: {age_group}")
    if gender not in (0, 1):
        raise ValueError(f"Invalid gender: {gender}")
    return age_group * 2 + gender


def add_labels(df: pd.DataFrame, age_col: str = "age", gender_col: str = "gender") -> pd.DataFrame:
    """Add age_group and final_label columns to a dataframe."""
    required = {age_col, gender_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["age_group"] = out[age_col].apply(age_to_group)
    out["final_label"] = [combined_label(a, g) for a, g in zip(out["age_group"], out[gender_col])]
    out["final_label"] = out["final_label"].astype(str)
    return out


def validate_image_paths(df: pd.DataFrame, path_col: str = "full_path") -> pd.DataFrame:
    """Return rows whose image path does not exist."""
    if path_col not in df.columns:
        raise KeyError(f"Missing path column: {path_col}")
    missing_rows = []
    for idx, value in df[path_col].items():
        if not Path(str(value)).exists():
            missing_rows.append({"row": idx, path_col: value})
    return pd.DataFrame(missing_rows)


def save_labeled_csv(input_csv: str | Path, output_csv: str | Path, age_col: str = "age", gender_col: str = "gender") -> Path:
    """Read a CSV, add age_group/final_label, and save it."""
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    df = pd.read_csv(input_csv)
    labeled = add_labels(df, age_col=age_col, gender_col=gender_col)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_csv, index=False)
    return output_csv


def parse_utkface_filename(filename: str) -> dict[str, int] | None:
    """Parse UTKFace filename format: age_gender_race_date.jpg.

    Returns None if the filename cannot be parsed.
    """
    name = Path(filename).name
    match = re.match(r"^(?P<age>\d+)_(?P<gender>[01])_(?P<race>\d+)_", name)
    if not match:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def create_utkface_csv(image_paths: Iterable[str | Path], output_csv: str | Path) -> Path:
    """Create a labeled CSV from UTKFace image filenames."""
    rows = []
    for path in image_paths:
        parsed = parse_utkface_filename(str(path))
        if parsed is None:
            continue
        rows.append({"full_path": str(path), "age": parsed["age"], "gender": parsed["gender"], "race": parsed["race"]})
    df = add_labels(pd.DataFrame(rows))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv
