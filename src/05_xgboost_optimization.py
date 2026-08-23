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
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
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


def load_data():
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

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def build_parameter_space():
    parameter_space = {
        "n_estimators": [
            300,
            500,
            700,
            900,
            1200,
        ],
        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.08,
            0.10,
        ],
        "max_depth": [
            3,
            4,
            5,
            6,
            7,
            8,
        ],
        "min_child_weight": [
            1,
            3,
            5,
            7,
        ],
        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "reg_alpha": [
            0,
            0.01,
            0.1,
            0.5,
            1.0,
        ],
        "reg_lambda": [
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ],
    }

    return parameter_space


def optimize_xgboost(
    X_train,
    y_train,
):
    print("\nSTEP 2 - XGBOOST HYPERPARAMETER SEARCH")

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    parameter_space = build_parameter_space()

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_space,
        n_iter=40,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    start_time = time.time()

    search.fit(
        X_train,
        y_train,
    )

    search_time = (
        time.time()
        - start_time
    )

    print(
        f"\nSearch time: "
        f"{search_time:.2f} seconds"
    )

    print("\nBest parameters:")

    for parameter, value in search.best_params_.items():
        print(f"{parameter}: {value}")

    best_cv_mae = -search.best_score_

    print(
        f"\nBest cross-validation MAE: "
        f"{best_cv_mae:.2f} K"
    )

    return search


def evaluate_best_model(
    search,
    X_train,
    X_test,
    y_train,
    y_test,
):
    print("\nSTEP 3 - FINAL TEST EVALUATION")

    best_model = search.best_estimator_

    train_predictions = best_model.predict(
        X_train
    )

    test_predictions = best_model.predict(
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

    results = pd.DataFrame(
        [
            {
                "Model": "Baseline XGBoost",
                "Train_R2": 0.9793,
                "Test_R2": 0.8835,
                "Test_MAE_K": 25.8590,
                "Test_RMSE_K": 38.4042,
            },
            {
                "Model": "Optimized XGBoost",
                "Train_R2": train_r2,
                "Test_R2": test_r2,
                "Test_MAE_K": test_mae,
                "Test_RMSE_K": test_rmse,
            },
        ]
    )

    return (
        best_model,
        test_predictions,
        results,
    )


def save_search_results(search):
    print("\nSTEP 4 - SAVE SEARCH RESULTS")

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    cv_results[
        "CV_MAE_K"
    ] = -cv_results[
        "mean_test_score"
    ]

    cv_results[
        "Train_MAE_K"
    ] = -cv_results[
        "mean_train_score"
    ]

    cv_results = cv_results.sort_values(
        "CV_MAE_K",
        ascending=True,
    )

    output_file = (
        TABLES_DIR
        / "xgboost_random_search_results.csv"
    )

    cv_results.to_csv(
        output_file,
        index=False,
    )

    print("\nSearch results saved to:")
    print(output_file)


def save_model_and_results(
    best_model,
    results,
):
    print("\nSTEP 5 - SAVE BEST MODEL")

    model_file = (
        MODELS_DIR
        / "xgboost_optimized.joblib"
    )

    joblib.dump(
        best_model,
        model_file,
    )

    results_file = (
        TABLES_DIR
        / "xgboost_optimization_comparison.csv"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    print("\nModel comparison:")
    print(
        results.round(4).to_string(
            index=False
        )
    )

    print("\nBest model saved to:")
    print(model_file)


def plot_predictions(
    y_test,
    predictions,
):
    print("\nSTEP 6 - CREATE PREDICTION FIGURE")

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.5,
    )

    minimum = min(
        y_test.min(),
        predictions.min(),
    )

    maximum = max(
        y_test.max(),
        predictions.max(),
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
        "Optimized XGBoost: "
        "Experimental vs Predicted Tg"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "xgboost_optimized_actual_vs_predicted.png"
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
    print("\nPOLYMER Tg XGBOOST OPTIMIZATION")

    create_output_directories()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()

    search = optimize_xgboost(
        X_train,
        y_train,
    )

    (
        best_model,
        predictions,
        results,
    ) = evaluate_best_model(
        search,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_search_results(
        search
    )

    save_model_and_results(
        best_model,
        results,
    )

    plot_predictions(
        y_test,
        predictions,
    )

    print("\nXGBOOST OPTIMIZATION COMPLETED")


if __name__ == "__main__":
    main()