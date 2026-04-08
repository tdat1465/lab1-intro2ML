import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.model_selection import KFold, learning_curve
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

REGISTRY_PATH = "results/metrics_registry.pkl"


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "rb") as f:
            return pickle.load(f)
    return {
        "test_errors_dict": {},
        "cv_scores_dict": {},
        "residuals_dict": {},
        "preds_dict": {},
        "metrics_list": [],
    }


def save_registry(data):
    # Đảm bảo folder exists
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "wb") as f:
        pickle.dump(data, f)


def update_registry(
    model_name, metric_dict, test_errors, cv_rmse_array, preds, residuals
):
    data = load_registry()
    data["test_errors_dict"][model_name] = test_errors
    data["cv_scores_dict"][model_name] = np.array(cv_rmse_array)
    data["preds_dict"][model_name] = preds
    data["residuals_dict"][model_name] = residuals

    # xoá model metric cũ nếu bị trùng tên (do chạy lại cell)
    data["metrics_list"] = [m for m in data["metrics_list"] if m["Model"] != model_name]
    data["metrics_list"].append(metric_dict)

    save_registry(data)


def load_bike_data():
    train_df = pd.read_csv(Path(__file__).parent / "../../data/regression/train.csv")
    val_df = pd.read_csv(Path(__file__).parent / "../../data/regression/val.csv")
    test_df = pd.read_csv(Path(__file__).parent / "../../data/regression/test.csv")

    features_to_drop = ["instant", "dteday", "casual", "registered"]

    def prepare_data(df):
        df = df.drop(columns=features_to_drop, errors="ignore")
        X = df.drop(columns=["cnt"]).values
        y = df["cnt"].values
        return X, y

    X_train, y_train = prepare_data(train_df)
    X_val, y_val = prepare_data(val_df)
    X_test, y_test = prepare_data(test_df)

    # Không chuẩn hoá do dữ liệu (temp, hum, windspeed...) đã được scale sẵn
    X_train_scaled = X_train
    X_val_scaled = X_val
    X_test_scaled = X_test

    X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_val_b = np.c_[np.ones((X_val_scaled.shape[0], 1)), X_val_scaled]
    X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

    return (
        train_df,
        features_to_drop,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        X_train_b,
        X_val_b,
        X_test_b,
    )


def evaluate_model(
    model_name,
    fit_predict_fn,
    X_tr_full,
    y_tr_full,
    X_te_full,
    y_te_full,
    X_va_full=None,
    y_va_full=None,
):
    if X_va_full is not None and y_va_full is not None:
        X_cv = np.vstack((X_tr_full, X_va_full))
        y_cv = np.concatenate((y_tr_full, y_va_full))
    else:
        X_cv = X_tr_full
        y_cv = y_tr_full

    kf = KFold(n_splits=10, shuffle=False)
    cv_mse = []
    cv_rmse = []
    cv_mae = []
    cv_r2 = []
    for train_index, val_index in kf.split(X_cv):
        X_fold_train, X_fold_val = X_cv[train_index], X_cv[val_index]
        y_fold_train, y_fold_val = y_cv[train_index], y_cv[val_index]

        y_val_pred, _ = fit_predict_fn(X_fold_train, y_fold_train, X_fold_val)

        mse = np.mean((y_fold_val - y_val_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_fold_val - y_val_pred))
        r2 = 1 - (
            np.sum((y_fold_val - y_val_pred) ** 2)
            / np.sum((y_fold_val - np.mean(y_fold_val)) ** 2)
        )
        cv_mse.append(mse)
        cv_rmse.append(rmse)
        cv_mae.append(mae)
        cv_r2.append(r2)

    mean_cv_mse = np.mean(cv_mse)
    std_cv_mse = np.std(cv_mse)
    mean_cv_rmse = np.mean(cv_rmse)
    std_cv_rmse = np.std(cv_rmse)
    mean_cv_mae = np.mean(cv_mae)
    std_cv_mae = np.std(cv_mae)
    mean_cv_r2 = np.mean(cv_r2)
    std_cv_r2 = np.std(cv_r2)

    if X_va_full is not None and y_va_full is not None:
        y_val_pred, _ = fit_predict_fn(X_cv, y_cv, X_va_full)
        val_mse = np.mean((y_va_full - y_val_pred) ** 2)
        val_rmse = np.sqrt(val_mse)
        val_mae = np.mean(np.abs(y_va_full - y_val_pred))
        val_r2 = 1 - (
            np.sum((y_va_full - y_val_pred) ** 2)
            / np.sum((y_va_full - np.mean(y_va_full)) ** 2)
        )
    else:
        val_mse, val_rmse, val_mae, val_r2 = None, None, None, None

    y_test_pred, final_model = fit_predict_fn(X_cv, y_cv, X_te_full)
    test_mse = np.mean((y_te_full - y_test_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_te_full - y_test_pred))
    test_r2 = 1 - (
        np.sum((y_te_full - y_test_pred) ** 2)
        / np.sum((y_te_full - np.mean(y_te_full)) ** 2)
    )

    abs_errors = np.abs(y_te_full - y_test_pred)
    residuals = y_te_full - y_test_pred

    metric_dict = {
        "Model": model_name,
        "CV_MSE (Mean ± Std)": f"{mean_cv_mse:.2f} ± {std_cv_mse:.2f}",
        "CV_RMSE (Mean ± Std)": f"{mean_cv_rmse:.2f} ± {std_cv_rmse:.2f}",
        "CV_MAE (Mean ± Std)": f"{mean_cv_mae:.2f} ± {std_cv_mae:.2f}",
        "CV_R2 (Mean ± Std)": f"{mean_cv_r2:.4f} ± {std_cv_r2:.4f}",
    }
    if val_mse is not None:
        metric_dict["Val MSE"] = val_mse
        metric_dict["Val RMSE"] = val_rmse
        metric_dict["Val MAE"] = val_mae
        metric_dict["Val R2"] = val_r2

    metric_dict.update(
        {
            "Test MSE": test_mse,
            "Test RMSE": test_rmse,
            "Test MAE": test_mae,
            "Test R2": test_r2,
        }
    )

    print(f"[{model_name}] Kết quả đánh giá:")
    for key, value in metric_dict.items():
        if key != "Model":
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")

    # Tự động vẽ Plot Actual/Residual ngay lập tức
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_te_full, y_test_pred, alpha=0.3, color="teal")
    axes[0].plot(
        [y_te_full.min(), y_te_full.max()],
        [y_te_full.min(), y_te_full.max()],
        "r--",
        lw=2,
    )
    axes[0].set_xlabel("Thực tế (Actual)")
    axes[0].set_ylabel("Dự đoán (Predicted)")
    axes[0].set_title(f"{model_name}: Actual vs Predicted")

    # Vẽ Scatter Plot kiểm tra tính ngẫu nhiên (homoscedasticity)
    residuals = y_te_full - y_test_pred
    axes[1].scatter(y_test_pred, residuals, alpha=0.3, color="coral")
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Dự đoán (Predicted)")
    axes[1].set_ylabel("Residuals (Thực tế - Dự đoán)")
    axes[1].set_title(f"{model_name}: Residuals vs Predicted")

    # Cất vào file
    update_registry(
        model_name, metric_dict, abs_errors, cv_rmse, y_test_pred, residuals
    )

    return final_model


def plot_learning_curve_data_size(
    model_name,
    fit_predict_fn,
    X_tr_full,
    y_tr_full,
    X_val_full,
    y_val_full,
    num_points=10,
):
    train_sizes = np.linspace(0.1, 1.0, num_points)
    train_mses = []
    val_mses = []

    for size in train_sizes:
        num_samples = int(size * len(X_tr_full))
        if num_samples == 0:
            continue

        X_tr_subset = X_tr_full[:num_samples]
        y_tr_subset = y_tr_full[:num_samples]

        y_tr_pred, _ = fit_predict_fn(X_tr_subset, y_tr_subset, X_tr_subset)
        train_mse = np.mean((y_tr_subset - y_tr_pred) ** 2)

        y_val_pred, _ = fit_predict_fn(X_tr_subset, y_tr_subset, X_val_full)
        val_mse = np.mean((y_val_full - y_val_pred) ** 2)

        train_mses.append(train_mse)
        val_mses.append(val_mse)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes * 100, train_mses, label="Train MSE", marker="o")
    plt.plot(train_sizes * 100, val_mses, label="Validation MSE", marker="s")
    plt.xlabel("Percentage of Training Data (%)")
    plt.ylabel("MSE")
    plt.title(f"Learning Curve (Theo kích thước dữ liệu) - Mẫu: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sklearn_learning_curve_wrapper(estimator, title, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training error")
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation error"
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )

    plt.title(f"Learning Curve (Train Size): {title}")
    plt.xlabel("Số lượng mẫu phân chia huấn luyện")
    plt.ylabel("RMSE (Thấp là tốt)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
