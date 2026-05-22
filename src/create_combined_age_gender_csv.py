import argparse
import os
import pandas as pd
import numpy as np


def age_to_fine_group(age):
    age = int(age)

    if age <= 13:
        return 0
    elif age <= 17:
        return 1
    elif age <= 24:
        return 2
    elif age <= 33:
        return 3
    elif age <= 48:
        return 4
    elif age <= 64:
        return 5
    else:
        return 6


def fine_to_coarse_group(fine_group):
    fine_group = int(fine_group)

    if fine_group == 0:
        return 0      # 0–13
    elif fine_group in [1,2,3,4]:
        return 1      # 14–48
    else:
        return 2      # 49+


def clean_gender(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["0", "0.0", "m", "male"]:
        return 0

    if value in ["1", "1.0", "f", "female"]:
        return 1

    return np.nan


def load_and_clean(csv_path, dataset_name):
    df = pd.read_csv(csv_path)

    if "full_path" not in df.columns:
        raise ValueError(f"{csv_path} must contain full_path column")

    if "age" not in df.columns:
        raise ValueError(f"{csv_path} must contain age column")

    if "gender" not in df.columns:
        raise ValueError(f"{csv_path} must contain gender column")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = df["gender"].apply(clean_gender)

    df = df.dropna(subset=["full_path", "age", "gender"])

    df["age"] = df["age"].astype(int)
    df["gender"] = df["gender"].astype(int)

    df = df[df["age"].between(0, 100)]
    df = df[df["gender"].between(0, 1)]

    df["fine_age_group"] = df["age"].apply(age_to_fine_group)
    df["coarse_age_group"] = df["fine_age_group"].apply(fine_to_coarse_group)

    df["dataset"] = dataset_name

    keep_cols = [
        "dataset",
        "full_path",
        "age",
        "gender",
        "coarse_age_group",
        "fine_age_group",
    ]

    optional_cols = [
        "quality_score",
        "blur_score",
        "brightness_score",
        "detection_confidence",
    ]

    for col in optional_cols:
        if col in df.columns:
            keep_cols.append(col)

    return df[keep_cols]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wiki-csv", required=True)
    parser.add_argument("--imdb-csv", required=True)
    parser.add_argument("--output-csv", required=True)

    args = parser.parse_args()

    wiki_df = load_and_clean(args.wiki_csv, "wiki")
    imdb_df = load_and_clean(args.imdb_csv, "imdb")

    combined = pd.concat([wiki_df, imdb_df], ignore_index=True)

    before = len(combined)
    combined = combined[combined["full_path"].apply(lambda x: os.path.exists(str(x)))]
    print(f"Valid image paths: {len(combined)} / {before}")

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nDataset distribution:")
    print(combined["dataset"].value_counts())

    print("\nGender distribution:")
    print(combined["gender"].value_counts().sort_index())

    print("\nCoarse age distribution:")
    print(combined["coarse_age_group"].value_counts().sort_index())

    print("\nFine age distribution:")
    print(combined["fine_age_group"].value_counts().sort_index())

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    combined.to_csv(args.output_csv, index=False)

    print(f"\nSaved combined CSV to: {args.output_csv}")
    print(f"Rows: {len(combined)}")


if __name__ == "__main__":
    main()