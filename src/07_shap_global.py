from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "model"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

MODEL_FILE = (
    MODELS_DIR
    / "xgboost_optimized.joblib"
)

TOP_N = 20


def create_output_directories():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_data():
    print("\nSTEP 1 - LOAD TEST DATA")

    X_test = pd.read_csv(
        MODEL_DATA_DIR / "X_test.csv"
    )

    print(f"\nX_test shape: {X_test.shape}")

    return X_test


def load_model():
    print("\nSTEP 2 - LOAD OPTIMIZED MODEL")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print("\nOptimized XGBoost loaded.")

    return model


def calculate_shap_values(
    model,
    X_test,
):
    print("\nSTEP 3 - CALCULATE SHAP VALUES")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X_test
    )

    print(
        f"\nSHAP values shape: "
        f"{shap_values.values.shape}"
    )

    return shap_values


def create_feature_importance_table(
    X_test,
    shap_values,
):
    print("\nSTEP 4 - SHAP FEATURE IMPORTANCE")

    mean_abs_shap = np.abs(
        shap_values.values
    ).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Mean_Abs_SHAP": mean_abs_shap,
        }
    )

    importance_df = importance_df.sort_values(
        "Mean_Abs_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    importance_df[
        "Rank"
    ] = np.arange(
        1,
        len(importance_df) + 1,
    )

    importance_df = importance_df[
        [
            "Rank",
            "Feature",
            "Mean_Abs_SHAP",
        ]
    ]

    output_file = (
        TABLES_DIR
        / "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        output_file,
        index=False,
    )

    print("\nTop 20 SHAP features:")

    print(
        importance_df.head(
            TOP_N
        ).to_string(index=False)
    )

    print("\nFeature importance saved to:")
    print(output_file)

    return importance_df


def plot_shap_bar(
    X_test,
    shap_values,
):
    print("\nSTEP 5 - CREATE SHAP BAR PLOT")

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=TOP_N,
        show=False,
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "shap_feature_importance.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSHAP bar plot saved to:")
    print(output_file)


def plot_shap_beeswarm(
    shap_values,
):
    print("\nSTEP 6 - CREATE SHAP BEESWARM")

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=TOP_N,
        show=False,
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "shap_beeswarm.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSHAP beeswarm saved to:")
    print(output_file)


def save_top_features(
    importance_df,
):
    print("\nSTEP 7 - SAVE TOP FEATURES")

    top_features = importance_df.head(
        TOP_N
    ).copy()

    output_file = (
        TABLES_DIR
        / "shap_top20_features.csv"
    )

    top_features.to_csv(
        output_file,
        index=False,
    )

    print("\nTop feature table saved to:")
    print(output_file)


def main():
    print("\nPOLYMER Tg SHAP GLOBAL ANALYSIS")

    create_output_directories()

    X_test = load_data()

    model = load_model()

    shap_values = calculate_shap_values(
        model,
        X_test,
    )

    importance_df = (
        create_feature_importance_table(
            X_test,
            shap_values,
        )
    )

    plot_shap_bar(
        X_test,
        shap_values,
    )

    plot_shap_beeswarm(
        shap_values,
    )

    save_top_features(
        importance_df
    )

    print("\nSHAP GLOBAL ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()