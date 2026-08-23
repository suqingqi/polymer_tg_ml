from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "polymer_tg_clean.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

TARGET_COLUMN = "labels.Exp_Tg(K)"

STRUCTURE_COLUMNS = [
    "PSMILES",
    "BIGSMILES",
]

CLASS_COLUMN = "meta.polymer_class"


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Clean data file not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    return df


def create_output_directories():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def inspect_dataset(df):
    print("\nSTEP 1 - DATASET OVERVIEW")

    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    non_numeric_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print(f"\nNumeric columns: {len(numeric_columns)}")
    print(
        f"Non-numeric columns: "
        f"{len(non_numeric_columns)}"
    )

    return numeric_columns, non_numeric_columns


def classify_features(df):
    print("\nSTEP 2 - FEATURE GROUP AUDIT")

    feature_groups = {
        "sidechain": [],
        "backbone": [],
        "fullpolymer": [],
        "other_numeric": [],
    }

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    for column in numeric_columns:
        if column == TARGET_COLUMN:
            continue

        column_lower = column.lower()

        if "sidechainlevel" in column_lower:
            feature_groups["sidechain"].append(column)

        elif "backbonelevel" in column_lower:
            feature_groups["backbone"].append(column)

        elif "fullpolymerlevel" in column_lower:
            feature_groups["fullpolymer"].append(column)

        else:
            feature_groups["other_numeric"].append(column)

    for group, columns in feature_groups.items():
        print(
            f"\n{group}: "
            f"{len(columns)} features"
        )

    audit_rows = []

    for group, columns in feature_groups.items():
        for column in columns:
            audit_rows.append(
                {
                    "feature": column,
                    "group": group,
                    "dtype": str(df[column].dtype),
                    "unique_values": df[column].nunique(),
                    "missing_values": df[column].isna().sum(),
                }
            )

    audit_df = pd.DataFrame(audit_rows)

    audit_file = (
        TABLES_DIR
        / "feature_audit.csv"
    )

    audit_df.to_csv(
        audit_file,
        index=False
    )

    print("\nFeature audit saved to:")
    print(audit_file)

    return feature_groups


def check_near_constant_features(
    df,
    feature_groups,
    threshold=0.99,
):
    print("\nSTEP 3 - NEAR-CONSTANT FEATURE CHECK")

    model_features = []

    for columns in feature_groups.values():
        model_features.extend(columns)

    near_constant_rows = []

    for column in model_features:
        value_counts = df[column].value_counts(
            normalize=True,
            dropna=False,
        )

        if len(value_counts) == 0:
            continue

        dominant_fraction = value_counts.iloc[0]

        if dominant_fraction >= threshold:
            near_constant_rows.append(
                {
                    "feature": column,
                    "dominant_fraction": dominant_fraction,
                    "unique_values": df[column].nunique(),
                }
            )

    near_constant_df = pd.DataFrame(
        near_constant_rows
    )

    if near_constant_df.empty:
        print(
            "\nNo near-constant features found "
            f"at threshold {threshold:.2f}."
        )

    else:
        near_constant_df = near_constant_df.sort_values(
            "dominant_fraction",
            ascending=False,
        )

        print(
            f"\nNear-constant features found: "
            f"{len(near_constant_df)}"
        )

        print(
            near_constant_df[
                [
                    "feature",
                    "dominant_fraction",
                    "unique_values",
                ]
            ].to_string(index=False)
        )

        output_file = (
            TABLES_DIR
            / "near_constant_features.csv"
        )

        near_constant_df.to_csv(
            output_file,
            index=False,
        )

        print("\nNear-constant feature table saved to:")
        print(output_file)

    return near_constant_df


def check_high_correlations(
    df,
    feature_groups,
    threshold=0.95,
):
    print("\nSTEP 4 - HIGH CORRELATION CHECK")

    model_features = []

    for columns in feature_groups.values():
        model_features.extend(columns)

    feature_df = df[model_features]

    correlation_matrix = feature_df.corr().abs()

    correlated_pairs = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            correlation = correlation_matrix.iloc[i, j]

            if correlation >= threshold:
                correlated_pairs.append(
                    {
                        "feature_1": columns[i],
                        "feature_2": columns[j],
                        "correlation": correlation,
                    }
                )

    correlation_df = pd.DataFrame(
        correlated_pairs
    )

    if correlation_df.empty:
        print(
            "\nNo highly correlated feature pairs "
            f"found above {threshold:.2f}."
        )

    else:
        correlation_df = correlation_df.sort_values(
            "correlation",
            ascending=False,
        )

        print(
            f"\nHighly correlated feature pairs: "
            f"{len(correlation_df)}"
        )

        output_file = (
            TABLES_DIR
            / "high_correlation_pairs.csv"
        )

        correlation_df.to_csv(
            output_file,
            index=False,
        )

        print("\nCorrelation table saved to:")
        print(output_file)

    return correlation_df


def plot_tg_distribution(df):
    print("\nSTEP 5 - Tg DISTRIBUTION")

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[TARGET_COLUMN],
        bins=40,
        edgecolor="black",
    )

    plt.xlabel("Glass Transition Temperature (K)")
    plt.ylabel("Count")
    plt.title("Distribution of Polymer Tg")

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "tg_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nTg distribution figure saved to:")
    print(output_file)


def plot_polymer_class_distribution(df):
    print("\nSTEP 6 - POLYMER CLASS DISTRIBUTION")

    if CLASS_COLUMN not in df.columns:
        print("\nPolymer class column not found.")
        return

    class_counts = df[
        CLASS_COLUMN
    ].value_counts()

    print(
        f"\nNumber of polymer classes: "
        f"{len(class_counts)}"
    )

    print("\nPolymer class counts:")
    print(class_counts)

    class_table = class_counts.reset_index()

    class_table.columns = [
        "polymer_class",
        "count",
    ]

    output_table = (
        TABLES_DIR
        / "polymer_class_counts.csv"
    )

    class_table.to_csv(
        output_table,
        index=False,
    )

    plt.figure(figsize=(10, 7))

    class_counts.sort_values().plot(
        kind="barh"
    )

    plt.xlabel("Number of Polymers")
    plt.ylabel("Polymer Class")
    plt.title("Polymer Class Distribution")

    plt.tight_layout()

    output_figure = (
        FIGURES_DIR
        / "polymer_class_distribution.png"
    )

    plt.savefig(
        output_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nPolymer class figure saved to:")
    print(output_figure)


def check_structure_duplicates(df):
    print("\nSTEP 7 - STRUCTURE DUPLICATE CHECK")

    for column in STRUCTURE_COLUMNS:
        if column not in df.columns:
            continue

        duplicated = df[column].duplicated().sum()

        unique = df[column].nunique()

        print(f"\nStructure column: {column}")
        print(f"Unique structures: {unique}")
        print(f"Duplicated structures: {duplicated}")


def main():
    print("\nPOLYMER Tg FEATURE AUDIT AND EDA")

    create_output_directories()

    df = load_data()

    inspect_dataset(df)

    feature_groups = classify_features(df)

    check_near_constant_features(
        df,
        feature_groups,
    )

    check_high_correlations(
        df,
        feature_groups,
    )

    plot_tg_distribution(df)

    plot_polymer_class_distribution(df)

    check_structure_duplicates(df)

    print("\nFEATURE AUDIT AND EDA COMPLETED")


if __name__ == "__main__":
    main()