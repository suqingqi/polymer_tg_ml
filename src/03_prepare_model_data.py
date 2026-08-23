from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "polymer_tg_clean.csv"
)

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "model"
)

TARGET_COLUMN = "labels.Exp_Tg(K)"

STRUCTURE_COLUMN = "PSMILES"

CLASS_COLUMN = "meta.polymer_class"

BIGSMILES_COLUMN = "BIGSMILES"

TEST_SIZE = 0.20

RANDOM_STATE = 42


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Clean data file not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    return df


def identify_model_features(df):
    print("\nSTEP 1 - IDENTIFY MODEL FEATURES")

    excluded_columns = [
        TARGET_COLUMN,
        STRUCTURE_COLUMN,
        CLASS_COLUMN,
        BIGSMILES_COLUMN,
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    print(f"\nModel features: {len(feature_columns)}")

    print("\nExcluded columns:")

    for column in excluded_columns:
        print(column)

    return feature_columns


def create_model_dataset(df, feature_columns):
    print("\nSTEP 2 - CREATE MODEL DATASET")

    X = df[feature_columns].copy()

    y = df[TARGET_COLUMN].copy()

    metadata = df[
        [
            STRUCTURE_COLUMN,
            CLASS_COLUMN,
            BIGSMILES_COLUMN,
        ]
    ].copy()

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Metadata shape: {metadata.shape}")

    print(f"\nX missing values: {X.isna().sum().sum()}")
    print(f"y missing values: {y.isna().sum()}")

    return X, y, metadata


def split_data(X, y, metadata):
    print("\nSTEP 3 - TRAIN TEST SPLIT")

    indices = X.index

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train = X.loc[train_indices].copy()
    X_test = X.loc[test_indices].copy()

    y_train = y.loc[train_indices].copy()
    y_test = y.loc[test_indices].copy()

    metadata_train = metadata.loc[
        train_indices
    ].copy()

    metadata_test = metadata.loc[
        test_indices
    ].copy()

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print(f"\nTraining percentage: {len(X_train) / len(X):.2%}")
    print(f"Test percentage: {len(X_test) / len(X):.2%}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
    )


def compare_target_distribution(y_train, y_test):
    print("\nSTEP 4 - TARGET DISTRIBUTION CHECK")

    summary = pd.DataFrame(
        {
            "train": [
                y_train.mean(),
                y_train.std(),
                y_train.min(),
                y_train.median(),
                y_train.max(),
            ],
            "test": [
                y_test.mean(),
                y_test.std(),
                y_test.min(),
                y_test.median(),
                y_test.max(),
            ],
        },
        index=[
            "mean",
            "std",
            "min",
            "median",
            "max",
        ],
    )

    print("\nTg distribution comparison:")
    print(summary.round(2))

    return summary


def check_class_distribution(
    metadata_train,
    metadata_test,
):
    print("\nSTEP 5 - POLYMER CLASS CHECK")

    train_classes = (
        metadata_train[CLASS_COLUMN]
        .value_counts(normalize=True)
    )

    test_classes = (
        metadata_test[CLASS_COLUMN]
        .value_counts(normalize=True)
    )

    class_comparison = pd.concat(
        [
            train_classes.rename("train_fraction"),
            test_classes.rename("test_fraction"),
        ],
        axis=1,
    ).fillna(0)

    class_comparison[
        "difference"
    ] = (
        class_comparison["train_fraction"]
        - class_comparison["test_fraction"]
    ).abs()

    class_comparison = (
        class_comparison
        .sort_values(
            "train_fraction",
            ascending=False,
        )
    )

    print("\nLargest class distribution differences:")

    print(
        class_comparison.head(10).round(4)
    )

    return class_comparison


def save_split_data(
    X_train,
    X_test,
    y_train,
    y_test,
    metadata_train,
    metadata_test,
    target_summary,
    class_comparison,
):
    print("\nSTEP 6 - SAVE MODEL DATA")

    MODEL_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train.to_csv(
        MODEL_DATA_DIR / "X_train.csv",
        index=False,
    )

    X_test.to_csv(
        MODEL_DATA_DIR / "X_test.csv",
        index=False,
    )

    y_train.to_csv(
        MODEL_DATA_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_csv(
        MODEL_DATA_DIR / "y_test.csv",
        index=False,
    )

    metadata_train.to_csv(
        MODEL_DATA_DIR / "metadata_train.csv",
        index=False,
    )

    metadata_test.to_csv(
        MODEL_DATA_DIR / "metadata_test.csv",
        index=False,
    )

    target_summary.to_csv(
        MODEL_DATA_DIR
        / "target_distribution_split.csv"
    )

    class_comparison.to_csv(
        MODEL_DATA_DIR
        / "class_distribution_split.csv"
    )

    print("\nModel data saved to:")
    print(MODEL_DATA_DIR)


def main():
    print("\nPOLYMER Tg MODEL DATA PREPARATION")

    df = load_data()

    feature_columns = identify_model_features(df)

    X, y, metadata = create_model_dataset(
        df,
        feature_columns,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
    ) = split_data(
        X,
        y,
        metadata,
    )

    target_summary = compare_target_distribution(
        y_train,
        y_test,
    )

    class_comparison = check_class_distribution(
        metadata_train,
        metadata_test,
    )

    save_split_data(
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test,
        target_summary,
        class_comparison,
    )

    print("\nMODEL DATA PREPARATION COMPLETED")


if __name__ == "__main__":
    main()