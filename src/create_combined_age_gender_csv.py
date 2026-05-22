import argparse
import os
import pandas as pd


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
    elif fine_group in [1, 2, 3, 4]:
        return 1      # 14–48
    else:
        return 2      # 49+


def load_dataset(csv_path, dataset_name):
    df = pd.read_csv(csv_path)

    required = [
        "full_path",
        "age",
        "gender"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    df = df.dropna(subset=required)

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce")

    df = df.dropna(subset=["age", "gender"])

    df["age"] = df["age"].astype(int)
    df["gender"] = df["gender"].astype(int)

    df = df[df["gender"].isin([0, 1])]

    before = len(df)

    df = df[df["full_path"].apply(lambda x: os.path.exists(str(x)))]

    print(f"\n{dataset_name}")
    print(f"Valid image paths: {len(df)} / {before}")

    df["fine_age_group"] = df["age"].apply(age_to_fine_group)

    df["coarse_age_group"] = df["fine_age_group"].apply(
        fine_to_coarse_group
    )

    df["dataset"] = dataset_name

    return df[
        [
            "dataset",
            "full_path",
            "age",
            "gender",
            "coarse_age_group",
            "fine_age_group",
        ]
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wiki-csv", required=True)
    parser.add_argument("--imdb-csv", required=True)
    parser.add_argument("--output-csv", required=True)

    args = parser.parse_args()

    wiki_df = load_dataset(
        args.wiki_csv,
        dataset_name="wiki"
    )

    imdb_df = load_dataset(
        args.imdb_csv,
        dataset_name="imdb"
    )

    combined_df = pd.concat(
        [wiki_df, imdb_df],
        ignore_index=True
    )

    combined_df = combined_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("\nCombined dataset size:")
    print(len(combined_df))

    print("\nFine age distribution:")
    print(
        combined_df["fine_age_group"]
        .value_counts()
        .sort_index()
    )

    print("\nCoarse age distribution:")
    print(
        combined_df["coarse_age_group"]
        .value_counts()
        .sort_index()
    )

    os.makedirs(
        os.path.dirname(args.output_csv),
        exist_ok=True
    )

    combined_df.to_csv(
        args.output_csv,
        index=False
    )

    print(f"\nSaved combined CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()