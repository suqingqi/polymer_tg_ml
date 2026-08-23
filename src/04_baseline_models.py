from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "model"
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

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

RANDOM_STATE = 42


def create_output_directories():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_model_data():
    print("\nSTEP 1 - LOAD MODEL DATA")

    X_train = pd.read_csv(
        MODEL_DATA_DIR / "X_train.csv"
    )

    X_test = pd.read_csv(
        MODEL_DATA_DIR / "X_test.csv"
    )

    y_train = pd.read_csv(
        MODEL_DATA_DIR / "y_train.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        MODEL_DATA_DIR / "y_test.csv"
    ).squeeze("columns")

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    print(f"\nTraining {model_name}")

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.time()
        - start_time
    )

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    train_r2 = r2_score(
        y_train,
        train_predictions,
    )

    test_r2 = r2_score(
        y_test,
        test_predictions,
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions,
        )
    )

    print(f"\nTrain R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} K")
    print(f"Test RMSE: {test_rmse:.2f} K")
    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    results = {
        "Model": model_name,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Test_MAE_K": test_mae,
        "Test_RMSE_K": test_rmse,
        "Training_Time_s": training_time,
    }

    return (
        results,
        test_predictions,
    )


def build_models():
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


def plot_actual_vs_predicted(
    y_test,
    predictions,
):
    print(
        "\nSTEP 3 - CREATE "
        "ACTUAL VS PREDICTED FIGURES"
    )

    for model_name, y_pred in predictions.items():

        plt.figure(
            figsize=(6, 6)
        )

        plt.scatter(
            y_test,
            y_pred,
            alpha=0.5,
        )

        minimum = min(
            y_test.min(),
            y_pred.min(),
        )

        maximum = max(
            y_test.max(),
            y_pred.max(),
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
        )

        plt.xlabel(
            "Experimental Tg (K)"
        )

        plt.ylabel(
            "Predicted Tg (K)"
        )

        plt.title(
            f"{model_name}: "
            f"Experimental vs Predicted Tg"
        )

        plt.tight_layout()

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        output_file = (
            FIGURES_DIR
            / f"{safe_name}_actual_vs_predicted.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\nSaved: {output_file}"
        )


def save_results(results):
    print("\nSTEP 4 - SAVE MODEL RESULTS")

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "Test_R2",
        ascending=False,
    )

    output_file = (
        TABLES_DIR
        / "baseline_model_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\nBaseline model comparison:")

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print("\nResults saved to:")
    print(output_file)

    return results_df


def main():
    print("\nPOLYMER Tg BASELINE MODELING")

    create_output_directories()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_model_data()

    models = build_models()

    print("\nSTEP 2 - TRAIN BASELINE MODELS")

    all_results = []

    all_predictions = {}

    for model_name, model in models.items():

        result, predictions = evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        all_results.append(
            result
        )

        all_predictions[
            model_name
        ] = predictions

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        joblib.dump(
            model,
            MODELS_DIR
            / f"{safe_name}_baseline.joblib",
        )

    plot_actual_vs_predicted(
        y_test,
        all_predictions,
    )

    save_results(
        all_results
    )

    print("\nBASELINE MODELING COMPLETED")


if __name__ == "__main__":
    main()