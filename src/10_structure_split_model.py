from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "model"
)

STRUCTURE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "structure_split"
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

OPTIMIZED_MODEL_FILE = (
    MODELS_DIR
    / "xgboost_optimized.joblib"
)


def load_split(data_dir):
    X_train = pd.read_csv(
        data_dir / "X_train.csv"
    )

    X_test = pd.read_csv(
        data_dir / "X_test.csv"
    )

    y_train = pd.read_csv(
        data_dir / "y_train.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        data_dir / "y_test.csv"
    ).squeeze("columns")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def load_optimized_parameters():
    print("\nSTEP 1 - LOAD OPTIMIZED PARAMETERS")

    if not OPTIMIZED_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Optimized model not found:\n"
            f"{OPTIMIZED_MODEL_FILE}"
        )

    model = joblib.load(
        OPTIMIZED_MODEL_FILE
    )

    parameters = model.get_params()

    print("\nOptimized parameters loaded.")

    return parameters


def train_and_evaluate(
    split_name,
    data_dir,
    parameters,
):
    print(
        f"\nSTEP 2 - TRAIN MODEL ON "
        f"{split_name.upper()} SPLIT"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_split(
        data_dir
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Test samples: "
        f"{len(X_test)}"
    )

    model = XGBRegressor(
        **parameters
    )

    model.fit(
        X_train,
        y_train,
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

    mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions,
        )
    )

    print(f"\nTrain R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MAE: {mae:.2f} K")
    print(f"Test RMSE: {rmse:.2f} K")

    result = {
        "Split": split_name,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Test_MAE_K": mae,
        "Test_RMSE_K": rmse,
    }

    return result


def compare_results(results):
    print("\nSTEP 3 - COMPARE SPLITS")

    results_df = pd.DataFrame(
        results
    )

    random_row = results_df[
        results_df["Split"]
        == "Random"
    ].iloc[0]

    cluster_row = results_df[
        results_df["Split"]
        == "Cluster-based"
    ].iloc[0]

    r2_drop = (
        random_row["Test_R2"]
        - cluster_row["Test_R2"]
    )

    mae_increase = (
        cluster_row["Test_MAE_K"]
        - random_row["Test_MAE_K"]
    )

    rmse_increase = (
        cluster_row["Test_RMSE_K"]
        - random_row["Test_RMSE_K"]
    )

    print("\nModel comparison:")

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print(
        f"\nR2 drop from Random to "
        f"Cluster-based: "
        f"{r2_drop:.4f}"
    )

    print(
        f"MAE increase: "
        f"{mae_increase:.2f} K"
    )

    print(
        f"RMSE increase: "
        f"{rmse_increase:.2f} K"
    )

    return results_df


def save_results(results_df):
    print("\nSTEP 4 - SAVE COMPARISON")

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        TABLES_DIR
        / "random_vs_cluster_split.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\nComparison saved to:")
    print(output_file)


def main():
    print(
        "\nPOLYMER Tg SPLIT COMPARISON"
    )

    parameters = (
        load_optimized_parameters()
    )

    random_result = (
        train_and_evaluate(
            "Random",
            RANDOM_DATA_DIR,
            parameters,
        )
    )

    cluster_result = (
        train_and_evaluate(
            "Cluster-based",
            STRUCTURE_DATA_DIR,
            parameters,
        )
    )

    results_df = compare_results(
        [
            random_result,
            cluster_result,
        ]
    )

    save_results(
        results_df
    )

    print(
        "\nSPLIT COMPARISON COMPLETED"
    )


if __name__ == "__main__":
    main()