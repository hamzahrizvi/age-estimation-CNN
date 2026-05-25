import argparse
import os
import pandas as pd
import numpy as np


def age_to_fine_group(age):
    age = int(age)

    if age <= 13:
        return 0      # 0-13
    elif age <= 17:
        return 1      # 14-17
    elif age <= 24:
        return 2      # 18-24
    elif age <= 33:
        return 3      # 25-33
    elif age <= 48:
        return 4      # 34-48
    elif age <= 64:
        return 5      # 49-64
    else:
        return 6      # 65+


def fine_to_coarse_group(fine_group):
    fine_group = int(fine_group)

    if fine_group == 0:
        return 0      # 0-13
    elif fine_group in [1, 2, 3, 4]:
        return 1      # 14-48
    else:
        return 2      # 49+


def clean_gender(value):
    """
    UTKFace uses:
    0 = male
    1 = female

    Our training standard is currently behaving as:
    0 = female
    1 = male

    So for UTK fine-tuning, flip:
    0 -> 1
    1 -> 0
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["0", "0.0", "m", "male"]:
        return 1

    if value in ["1", "1.0", "f", "female"]:
        return 0

    return np.nan


def find_image_column(df):
    candidates = [
        "full_path",
        "image_path",
        "path",
        "file_path",
        "filename",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find image path column. Expected one of: "
        "full_path, image_path, path, file_path, filename"
    )


def resolve_path(value, image_root=None):
    value = str(value).strip()

    if value and value.lower() not in ["nan", "none", ""]:
        if os.path.exists(value):
            return value

        if image_root is not None:
            candidate = os.path.join(image_root, value)
            if os.path.exists(candidate):
                return candidate

            candidate = os.path.join(image_root, os.path.basename(value))
            if os.path.exists(candidate):
                return candidate

    return None


def parse_utk_filename(filename):
    """
    UTKFace filenames are usually:
    age_gender_race_date.jpg

    Example:
    25_0_2_20170116174525125.jpg
    """
    base = os.path.basename(str(filename))
    parts = base.split("_")

    if len(parts) < 2:
        return None, None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        return age, gender
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--image-root", default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    print("Input columns:")
    print(list(df.columns))

    image_col = find_image_column(df)

    print(f"Using image column: {image_col}")

    df = df.copy()

    if "age" not in df.columns or "gender" not in df.columns:
        print("age/gender columns missing or incomplete. Parsing from filename where needed...")

        parsed = df[image_col].apply(parse_utk_filename)
        df["parsed_age"] = parsed.apply(lambda x: x[0])
        df["parsed_gender"] = parsed.apply(lambda x: x[1])

        if "age" not in df.columns:
            df["age"] = df["parsed_age"]
        else:
            df["age"] = df["age"].fillna(df["parsed_age"])

        if "gender" not in df.columns:
            df["gender"] = df["parsed_gender"]
        else:
            df["gender"] = df["gender"].fillna(df["parsed_gender"])

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = df["gender"].apply(clean_gender)

    df = df.dropna(subset=["age", "gender"])

    df["age"] = df["age"].astype(int)
    df["gender"] = df["gender"].astype(int)

    df = df[df["age"].between(0, 100)]
    df = df[df["gender"].between(0, 1)]

    print("Resolving image paths...")

    df["full_path"] = df[image_col].apply(
        lambda x: resolve_path(x, args.image_root)
    )

    before = len(df)
    df = df.dropna(subset=["full_path"])
    print(f"Valid image paths: {len(df)} / {before}")

    df["fine_age_group"] = df["age"].apply(age_to_fine_group)
    df["coarse_age_group"] = df["fine_age_group"].apply(fine_to_coarse_group)

    keep_cols = [
        "full_path",
        "age",
        "gender",
        "coarse_age_group",
        "fine_age_group",
    ]

    optional_cols = [
        "race",
        "quality_score",
        "blur_score",
        "brightness_score",
        "detection_confidence",
    ]

    for col in optional_cols:
        if col in df.columns:
            keep_cols.append(col)

    final_df = df[keep_cols].copy()
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nFine age distribution:")
    print(final_df["fine_age_group"].value_counts().sort_index())

    print("\nCoarse age distribution:")
    print(final_df["coarse_age_group"].value_counts().sort_index())

    print("\nGender distribution:")
    print(final_df["gender"].value_counts().sort_index())

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)

    print(f"\nSaved UTK hierarchical CSV to: {args.output_csv}")
    print(f"Rows: {len(final_df)}")


if __name__ == "__main__":
    main()