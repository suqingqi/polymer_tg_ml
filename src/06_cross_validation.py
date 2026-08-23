from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold


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

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

MODEL_FILE = (
    MODELS_DIR
    / "xgboost_optimized.joblib"
)

RANDOM_STATE = 42

N_SPLITS = 5


def create_output_directories():
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_data():
    print("\nSTEP 1 - LOAD TRAINING DATA")

    X_train = pd.read_csv(
        MODEL_DATA_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        MODEL_DATA_DIR / "y_train.csv"
    ).squeeze("columns")

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    return X_train, y_train


def load_model():
    print("\nSTEP 2 - LOAD OPTIMIZED MODEL")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Optimized model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print("\nOptimized XGBoost loaded.")

    return model


def run_cross_validation(
    model,
    X,
    y,
):
    print("\nSTEP 3 - RUN 5-FOLD CROSS VALIDATION")

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        kfold.split(X),
        start=1,
    ):

        print(f"\nFold {fold}")

        X_fold_train = X.iloc[
            train_index
        ]

        X_fold_validation = X.iloc[
            validation_index
        ]

        y_fold_train = y.iloc[
            train_index
        ]

        y_fold_validation = y.iloc[
            validation_index
        ]

        fold_model = model.__class__(
            **model.get_params()
        )

        start_time = time.time()

        fold_model.fit(
            X_fold_train,
            y_fold_train,
        )

        training_time = (
            time.time()
            - start_time
        )

        predictions = fold_model.predict(
            X_fold_validation
        )

        r2 = r2_score(
            y_fold_validation,
            predictions,
        )

        mae = mean_absolute_error(
            y_fold_validation,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_fold_validation,
                predictions,
            )
        )

        print(f"R2: {r2:.4f}")
        print(f"MAE: {mae:.2f} K")
        print(f"RMSE: {rmse:.2f} K")

        fold_results.append(
            {
                "Fold": fold,
                "R2": r2,
                "MAE_K": mae,
                "RMSE_K": rmse,
                "Training_Time_s": training_time,
            }
        )

    return pd.DataFrame(
        fold_results
    )


def summarize_results(results):
    print("\nSTEP 4 - CROSS-VALIDATION SUMMARY")

    mean_r2 = results["R2"].mean()
    std_r2 = results["R2"].std()

    mean_mae = results["MAE_K"].mean()
    std_mae = results["MAE_K"].std()

    mean_rmse = results["RMSE_K"].mean()
    std_rmse = results["RMSE_K"].std()

    print(
        f"\nR2: "
        f"{mean_r2:.4f} ± {std_r2:.4f}"
    )

    print(
        f"MAE: "
        f"{mean_mae:.2f} ± "
        f"{std_mae:.2f} K"
    )

    print(
        f"RMSE: "
        f"{mean_rmse:.2f} ± "
        f"{std_rmse:.2f} K"
    )

    summary = pd.DataFrame(
        [
            {
                "Metric": "R2",
                "Mean": mean_r2,
                "Std": std_r2,
            },
            {
                "Metric": "MAE_K",
                "Mean": mean_mae,
                "Std": std_mae,
            },
            {
                "Metric": "RMSE_K",
                "Mean": mean_rmse,
                "Std": std_rmse,
            },
        ]
    )

    return summary


def save_results(
    fold_results,
    summary,
):
    print("\nSTEP 5 - SAVE RESULTS")

    fold_file = (
        TABLES_DIR
        / "xgboost_cv_fold_results.csv"
    )

    summary_file = (
        TABLES_DIR
        / "xgboost_cv_summary.csv"
    )

    fold_results.to_csv(
        fold_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print("\nFold results saved to:")
    print(fold_file)

    print("\nCV summary saved to:")
    print(summary_file)


def plot_cv_results(results):
    print("\nSTEP 6 - CREATE CV FIGURE")

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        results["Fold"].astype(str),
        results["R2"],
    )

    plt.xlabel("Fold")
    plt.ylabel("R2")
    plt.title(
        "XGBoost 5-Fold Cross-Validation"
    )

    plt.ylim(
        max(0, results["R2"].min() - 0.05),
        min(1, results["R2"].max() + 0.05),
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "xgboost_cross_validation_r2.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nFigure saved to:")
    print(output_file)


def main():
    print("\nPOLYMER Tg XGBOOST CROSS VALIDATION")

    create_output_directories()

    X_train, y_train = load_data()

    model = load_model()

    fold_results = run_cross_validation(
        model,
        X_train,
        y_train,
    )

    summary = summarize_results(
        fold_results
    )

    save_results(
        fold_results,
        summary,
    )

    plot_cv_results(
        fold_results
    )

    print("\nCROSS VALIDATION COMPLETED")


if __name__ == "__main__":
    main()