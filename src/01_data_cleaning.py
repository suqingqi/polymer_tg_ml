from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "polymer_tg_raw.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "polymer_tg_clean.csv"

TARGET_COLUMN = "labels.Exp_Tg(K)"


def load_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    return df


def basic_data_check(df):
    print("\nSTEP 1 - BASIC DATA CHECK")

    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn data types:")
    print(df.dtypes.value_counts())

    total_missing = df.isna().sum().sum()

    print(f"\nTotal missing values: {total_missing}")

    duplicated_rows = df.duplicated().sum()

    print(f"Duplicated rows: {duplicated_rows}")


def check_target(df):
    print("\nSTEP 2 - TARGET CHECK")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column not found: {TARGET_COLUMN}"
        )

    print(f"\nTarget column: {TARGET_COLUMN}")

    print("\nTarget statistics:")
    print(df[TARGET_COLUMN].describe())

    target_missing = df[TARGET_COLUMN].isna().sum()

    print(f"\nTarget missing values: {target_missing}")


def identify_feature_types(df):
    print("\nSTEP 3 - FEATURE TYPE CHECK")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    non_numeric_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print(f"\nNumeric columns: {len(numeric_columns)}")
    print(f"Non-numeric columns: {len(non_numeric_columns)}")

    print("\nNon-numeric columns:")

    for column in non_numeric_columns:
        print(column)

    return numeric_columns, non_numeric_columns


def remove_duplicate_rows(df):
    print("\nSTEP 4 - DUPLICATE CHECK")

    before = len(df)

    df = df.drop_duplicates().copy()

    after = len(df)

    removed = before - after

    print(f"\nDuplicated rows removed: {removed}")
    print(f"Rows remaining: {after}")

    return df


def remove_constant_features(df):
    print("\nSTEP 5 - CONSTANT FEATURE CHECK")

    constant_columns = []

    for column in df.columns:
        if column == TARGET_COLUMN:
            continue

        if df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)

    if constant_columns:
        print("\nConstant features found:")

        for column in constant_columns:
            print(column)

        df = df.drop(
            columns=constant_columns
        )

        print(
            f"\nConstant features removed: "
            f"{len(constant_columns)}"
        )

    else:
        print("\nNo constant features found.")

    return df, constant_columns


def check_missing_values(df):
    print("\nSTEP 6 - MISSING VALUE CHECK")

    missing_values = df.isna().sum()

    missing_values = missing_values[
        missing_values > 0
    ].sort_values(ascending=False)

    if missing_values.empty:
        print("\nNo missing values found.")

    else:
        print("\nColumns containing missing values:")
        print(missing_values)


def check_target_range(df):
    print("\nSTEP 7 - TARGET RANGE CHECK")

    minimum_tg = df[TARGET_COLUMN].min()
    maximum_tg = df[TARGET_COLUMN].max()
    mean_tg = df[TARGET_COLUMN].mean()
    median_tg = df[TARGET_COLUMN].median()

    print(f"\nMinimum Tg: {minimum_tg:.2f} K")
    print(f"Maximum Tg: {maximum_tg:.2f} K")
    print(f"Mean Tg: {mean_tg:.2f} K")
    print(f"Median Tg: {median_tg:.2f} K")


def save_clean_data(df):
    print("\nSTEP 8 - SAVE CLEAN DATA")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nClean data saved to:")
    print(OUTPUT_FILE)

    print(
        f"\nFinal dataset shape: "
        f"{df.shape[0]} rows, "
        f"{df.shape[1]} columns"
    )


def main():
    print("\nPOLYMER Tg DATA CLEANING")

    df = load_data(RAW_FILE)

    basic_data_check(df)

    check_target(df)

    identify_feature_types(df)

    df = remove_duplicate_rows(df)

    df, constant_columns = remove_constant_features(df)

    check_missing_values(df)

    check_target_range(df)

    save_clean_data(df)

    print("\nDATA CLEANING COMPLETED")


if __name__ == "__main__":
    main()