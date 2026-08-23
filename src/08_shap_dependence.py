from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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

MODEL_FILE = (
    MODELS_DIR
    / "xgboost_optimized.joblib"
)


TOP_FEATURES = [
    "fullpolymerlevel.features.num_rings_sum_fullpolymerfeaturizer",

    "fullpolymerlevel.features.num_rotatable_bonds_sum_fullpolymerfeaturizer",

    "backbonelevel.features.num_rings_sum_backbonefeaturizer",

    "fullpolymerlevel.features.num_hbond_donors_sum_fullpolymerfeaturizer",

    "backbonelevel.features.sp2_carbon_count_sum_backbonefeaturizer",

    "backbonelevel.features.num_hbond_donors_sum_backbonefeaturizer",
]


FEATURE_NAMES = {
    "fullpolymerlevel.features.num_rings_sum_fullpolymerfeaturizer":
        "Full Polymer Ring Count",

    "fullpolymerlevel.features.num_rotatable_bonds_sum_fullpolymerfeaturizer":
        "Full Polymer Rotatable Bonds",

    "backbonelevel.features.num_rings_sum_backbonefeaturizer":
        "Backbone Ring Count",

    "fullpolymerlevel.features.num_hbond_donors_sum_fullpolymerfeaturizer":
        "Full Polymer H-Bond Donors",

    "backbonelevel.features.sp2_carbon_count_sum_backbonefeaturizer":
        "Backbone sp2 Carbon Count",

    "backbonelevel.features.num_hbond_donors_sum_backbonefeaturizer":
        "Backbone H-Bond Donors",
}


def create_output_directory():
    FIGURES_DIR.mkdir(
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


def create_dependence_plots(
    X_test,
    shap_values,
):
    print("\nSTEP 4 - CREATE DEPENDENCE PLOTS")

    for feature in TOP_FEATURES:

        if feature not in X_test.columns:
            print(
                f"\nFeature not found: "
                f"{feature}"
            )
            continue

        feature_index = X_test.columns.get_loc(
            feature
        )

        feature_shap = shap_values[
            :,
            feature_index
        ]

        plt.figure(
            figsize=(7, 5)
        )

        plt.scatter(
            X_test[feature],
            feature_shap.values,
            alpha=0.5,
        )

        plt.axhline(
            y=0,
            linewidth=1,
        )

        plt.xlabel(
            FEATURE_NAMES.get(
                feature,
                feature,
            )
        )

        plt.ylabel(
            "SHAP Value for Tg Prediction"
        )

        plt.title(
            f"SHAP Dependence: "
            f"{FEATURE_NAMES.get(feature, feature)}"
        )

        plt.tight_layout()

        safe_name = (
            FEATURE_NAMES.get(
                feature,
                feature,
            )
            .lower()
            .replace(" ", "_")
            .replace("-", "")
        )

        output_file = (
            FIGURES_DIR
            / f"shap_dependence_{safe_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\nSaved: "
            f"{output_file}"
        )


def create_feature_value_summary(
    X_test,
):
    print("\nSTEP 5 - FEATURE VALUE SUMMARY")

    available_features = [
        feature
        for feature in TOP_FEATURES
        if feature in X_test.columns
    ]

    summary = (
        X_test[
            available_features
        ]
        .describe()
        .T
    )

    summary.index = [
        FEATURE_NAMES.get(
            feature,
            feature,
        )
        for feature in summary.index
    ]

    print("\nTop feature statistics:")

    print(
        summary[
            [
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ].round(3)
    )


def main():
    print("\nPOLYMER Tg SHAP DEPENDENCE ANALYSIS")

    create_output_directory()

    X_test = load_data()

    model = load_model()

    shap_values = calculate_shap_values(
        model,
        X_test,
    )

    create_dependence_plots(
        X_test,
        shap_values,
    )

    create_feature_value_summary(
        X_test
    )

    print("\nSHAP DEPENDENCE ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()