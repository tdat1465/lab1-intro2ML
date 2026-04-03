# %% [markdown]
# # Phân tích Hồi quy (Regression) trên tập dữ liệu Bike Sharing
#
# Cài đặt các thuật toán học máy từ đầu (Numpy) cho phần Linear Regression và Regularization, và sử dụng Sklearn cho các hàm nâng cao.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

# Lưu trữ kết quả kiểm định, test prediction của mọi mô hình
test_errors_dict = {}
cv_scores_dict = {}
residuals_dict = {}
preds_dict = {}
metrics_list = []

# %% [markdown]
# ## 1. Đọc và Tiền Xử Lý Dữ Liệu
# %%
train_df = pd.read_csv("../../data/train.csv")
val_df = pd.read_csv("../../data/val.csv")
test_df = pd.read_csv("../../data/test.csv")

features_to_drop = ["instant", "dteday", "casual", "registered"]


def prepare_data(df):
    df = df.drop(columns=features_to_drop, errors="ignore")
    X = df.drop(columns=["cnt"]).values
    y = df["cnt"].values
    return X, y


X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)
X_test, y_test = prepare_data(test_df)

# Chuẩn hoá dữ liệu (Z-score normalization)
mean_feat = np.mean(X_train, axis=0)
std_feat = np.std(X_train, axis=0) + 1e-8
X_train_scaled = (X_train - mean_feat) / std_feat
X_val_scaled = (X_val - mean_feat) / std_feat
X_test_scaled = (X_test - mean_feat) / std_feat

X_train_scaled_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_val_scaled_b = np.c_[np.ones((X_val_scaled.shape[0], 1)), X_val_scaled]
X_test_scaled_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]


# Cơ chế K-Fold CV + Test Evaluation tự động cho tất cả các mô hình
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
    # Combine Train and Val for K-Fold CV if Val is provided
    if X_va_full is not None and y_va_full is not None:
        X_cv = np.vstack((X_tr_full, X_va_full))
        y_cv = np.concatenate((y_tr_full, y_va_full))
    else:
        X_cv = X_tr_full
        y_cv = y_tr_full

    # 1. 10-Fold CV trên tập Train+Val (Dùng Sklearn Library)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_mse = []
    cv_r2 = []
    for train_index, val_index in kf.split(X_cv):
        X_fold_train, X_fold_val = X_cv[train_index], X_cv[val_index]
        y_fold_train, y_fold_val = y_cv[train_index], y_cv[val_index]

        y_val_pred = fit_predict_fn(X_fold_train, y_fold_train, X_fold_val)

        mse = np.mean((y_fold_val - y_val_pred) ** 2)
        r2 = 1 - (
            np.sum((y_fold_val - y_val_pred) ** 2)
            / np.sum((y_fold_val - np.mean(y_fold_val)) ** 2)
        )
        cv_mse.append(mse)
        cv_r2.append(r2)

    mean_cv_mse = np.mean(cv_mse)
    std_cv_mse = np.std(cv_mse)
    mean_cv_r2 = np.mean(cv_r2)
    std_cv_r2 = np.std(cv_r2)

    cv_rmse = np.sqrt(cv_mse)
    cv_scores_dict[model_name] = cv_rmse

    # 1.5 Validation Set Evaluation
    if X_va_full is not None and y_va_full is not None:
        y_val_pred = fit_predict_fn(X_cv, y_cv, X_va_full)
        val_mse = np.mean((y_va_full - y_val_pred) ** 2)
        val_rmse = np.sqrt(val_mse)
        val_r2 = 1 - (
            np.sum((y_va_full - y_val_pred) ** 2)
            / np.sum((y_va_full - np.mean(y_va_full)) ** 2)
        )
    else:
        val_mse, val_rmse, val_r2 = None, None, None

    # 2. Test Set Evaluation
    y_test_pred = fit_predict_fn(X_cv, y_cv, X_te_full)
    test_mse = np.mean((y_te_full - y_test_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_te_full - y_test_pred))
    test_r2 = 1 - (
        np.sum((y_te_full - y_test_pred) ** 2)
        / np.sum((y_te_full - np.mean(y_te_full)) ** 2)
    )

    # Save absolute errors for Wilcoxon Rank Sum test sau cùng
    abs_errors = np.abs(y_te_full - y_test_pred)
    test_errors_dict[model_name] = abs_errors
    residuals_dict[model_name] = y_te_full - y_test_pred
    preds_dict[model_name] = y_test_pred

    # 3. Thêm Metric và In ra kết quả trực tiếp
    metric_dict = {
        "Model": model_name,
        "CV_MSE (Mean ± Std)": f"{mean_cv_mse:.2f} ± {std_cv_mse:.2f}",
        "CV_R2 (Mean ± Std)": f"{mean_cv_r2:.4f} ± {std_cv_r2:.4f}",
    }
    if val_mse is not None:
        metric_dict["Val MSE"] = val_mse
        metric_dict["Val RMSE"] = val_rmse
        metric_dict["Val R2"] = val_r2

    metric_dict.update(
        {
            "Test MSE": test_mse,
            "Test RMSE": test_rmse,
            "Test MAE": test_mae,
            "Test R2": test_r2,
        }
    )
    metrics_list.append(metric_dict)

    print(f"[{model_name}] Kết quả đánh giá:")
    for key, value in metric_dict.items():
        if key != "Model":
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")

    # 4. Tự động vẽ Plot Actual/Residual ngay lập tức
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

    plt.tight_layout()
    plt.show()


# Helper Hàm vẽ Learning Curve theo lượng data (Data Size)
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
    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset_len = int(size * len(X_tr_full))
        subset_len = max(10, subset_len)  # Đảm bảo có ít nhất 10 mẫu để fit
        X_sub = X_tr_full[:subset_len]
        y_sub = y_tr_full[:subset_len]

        y_train_pred = fit_predict_fn(X_sub, y_sub, X_sub)
        y_val_pred = fit_predict_fn(X_sub, y_sub, X_val_full)

        train_mse = np.mean((y_sub - y_train_pred) ** 2)
        val_mse = np.mean((y_val_full - y_val_pred) ** 2)

        train_scores.append(train_mse)
        val_scores.append(val_mse)

    plt.figure(figsize=(10, 6))
    plt.plot(
        train_sizes * len(X_tr_full),
        train_scores,
        label="Train MSE",
        marker="o",
        color="teal",
    )
    plt.plot(
        train_sizes * len(X_tr_full),
        val_scores,
        label="Validation MSE",
        marker="x",
        color="coral",
    )
    plt.xlabel("Số lượng mẫu huấn luyện (Training Set Size)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title(f"Learning Curve vs Mẫu Data ({model_name})")
    plt.legend()
    plt.show()


# %% [markdown]
# ## 2. Hồi quy Tuyến Tính (Thuần Numpy)
# ### 2.1 Normal Equations
# %%
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]


def fit_pred_ols(X_tr, y_tr, X_val):
    w = np.linalg.pinv(X_tr.T.dot(X_tr)).dot(X_tr.T).dot(y_tr)
    return X_val.dot(w)


evaluate_model(
    "Normal Equations OLS",
    fit_pred_ols,
    X_train_b,
    y_train,
    X_test_b,
    y_test,
    X_val_b,
    y_val,
)

# %% [markdown]
# ### 2.1.1 Kiểm tra giả định Gauss-Markov
# Kiểm tra QQ-plot xác định phân phối chuẩn và Breusch-Pagan test kiểm tra Heteroscedasticity.
# %%
y_train_pred_ols = fit_pred_ols(X_train_b, y_train, X_train_b)
residuals_train_ols = y_train - y_train_pred_ols

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Kiểm tra tính chuẩn (Normality) qua QQ-Plot trên Train
stats.probplot(residuals_train_ols, dist="norm", plot=axes[0])
axes[0].set_title("QQ Plot phần dư OLS (Train Set)")

# 2. Kiểm tra phương sai sai số không đổi (Homoscedasticity) qua Residual Plot trên Train
axes[1].scatter(y_train_pred_ols, residuals_train_ols, alpha=0.4, color="coral")
axes[1].axhline(y=0, color="red", linestyle="--", lw=2)
axes[1].set_xlabel("Giá trị dự đoán (Predicted - Train)")
axes[1].set_ylabel("Phần dư (Residuals - Train)")
axes[1].set_title("Residual Plot OLS (Train Set)")

plt.tight_layout()
plt.show()


def breusch_pagan_test(X, residuals):
    squared_residuals = residuals**2
    w_bp = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(squared_residuals)
    pred_squared_residuals = X.dot(w_bp)

    ss_res = np.sum((squared_residuals - pred_squared_residuals) ** 2)
    ss_tot = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)
    r2_bp = 1 - (ss_res / ss_tot)

    n = X.shape[0]
    p = X.shape[1] - 1
    lm_stat = n * r2_bp
    p_value = stats.chi2.sf(lm_stat, p)
    return lm_stat, p_value


bp_stat, bp_pval = breusch_pagan_test(X_train_b, residuals_train_ols)
print(f"Breusch-Pagan Test (Train): LM Stat = {bp_stat:.4f}, p-value = {bp_pval:.4e}")
if bp_pval < 0.05:
    print(
        "Có hiện tượng heteroscedasticity (phương sai sai số thay đổi). Nên dùng WLS."
    )
else:
    print("Không có hiện tượng heteroscedasticity.")


# %% [markdown]
# ### 2.2 Mini-batch Gradient Descent
# Biểu đồ Learning curve của hệ thống huấn luyện lặp theo kỉ nguyên
# %%
def minibatch_gd(X, y, epochs=100, batch_size=128, lr_init=0.01):
    m, n_features = X.shape
    np.random.seed(42)
    w = np.random.randn(n_features) * 0.01
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        lr = lr_init * 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        epoch_loss = 0
        for i in range(0, m, batch_size):
            X_i = X_shuffled[i : i + batch_size]
            y_i = y_shuffled[i : i + batch_size]

            predictions = X_i.dot(w)
            error = predictions - y_i
            epoch_loss += np.sum(error**2)

            gradient = (1 / batch_size) * X_i.T.dot(error)
            w -= lr * gradient

        losses.append(epoch_loss / m)

    return w, losses


# Lấy w_gd dùng sau nếu cần (hoặc chỉ quan sát quá trình chạy cơ bản)
w_gd, _ = minibatch_gd(
    X_train_scaled_b, y_train, epochs=200, batch_size=128, lr_init=0.05
)


def fit_pred_gd(X_tr, y_tr, X_val):
    w, _ = minibatch_gd(X_tr, y_tr, epochs=200, batch_size=128, lr_init=0.05)
    return X_val.dot(w)


evaluate_model(
    "Mini-Batch GD",
    fit_pred_gd,
    X_train_scaled_b,
    y_train,
    X_test_scaled_b,
    y_test,
    X_val_scaled_b,
    y_val,
)

plot_learning_curve_data_size(
    "Mini-Batch GD", fit_pred_gd, X_train_scaled_b, y_train, X_val_scaled_b, y_val
)

# %% [markdown]
# ### So sánh thời gian hội tụ (Training Time) giữa OLS và Mini-Batch GD
# Đo lường tốc độ thực thi thuần tuý của công thức đại số tuyến tính so với vòng lặp tối ưu Gradient Descent.
# %%
import time

# 1. Đo thời gian Normal Equations (OLS)
start_ols = time.process_time()
_ = np.linalg.pinv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
time_ols = time.process_time() - start_ols

# 2. Đo thời gian Mini-Batch Gradient Descent
start_gd = time.process_time()
_, _ = minibatch_gd(X_train_scaled_b, y_train, epochs=200, batch_size=128, lr_init=0.05)
time_gd = time.process_time() - start_gd

print("=" * 60)
print("SO SÁNH THỜI GIAN HUẤN LUYỆN (OLS vs Mini-Batch GD)")
print("=" * 60)
print(f"Thời gian OLS (Normal Equations): {time_ols:.5f} giây")
print(f"Thời gian Mini-Batch GD (200 epochs): {time_gd:.5f} giây")

if time_ols < time_gd:
    print(
        f"=> OLS hội tụ nhanh hơn {time_gd / time_ols:.2f} lần so với Mini-Batch GD trên tập dữ liệu này."
    )
else:
    print(f"=> Mini-Batch GD hội tụ nhanh hơn {time_ols / time_gd:.2f} lần so với OLS.")

# Vẽ biểu đồ cột để minh hoạ trực quan
plt.figure(figsize=(6, 4))
sns.barplot(
    x=["OLS (Normal Equations)", "Mini-Batch GD"], y=[time_ols, time_gd], palette="Set2"
)
plt.ylabel("Thời gian chạy (giây)")
plt.title("So sánh kích thước thời gian huấn luyện")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 2.3 WLS (Feasible Generalized Least Squares)
# %%
def fit_pred_wls(X_tr, y_tr, X_val):
    # Bước 1: Fit OLS thông thường
    w_ols = np.linalg.pinv(X_tr.T.dot(X_tr)).dot(X_tr.T).dot(y_tr)
    y_pred_tr = X_tr.dot(w_ols)
    residuals_abs = np.abs(y_tr - y_pred_tr)

    # Bước 2: Hồi quy phần dư tuyệt đối theo X để dự đoán phương sai phân phối tại từng điểm (Auxiliary FGLS)
    w_aux = np.linalg.pinv(X_tr.T.dot(X_tr)).dot(X_tr.T).dot(residuals_abs)
    volatility_est = X_tr.dot(w_aux)

    # Cắt các giá trị dự đoán rủi ro âm/gần 0
    volatility_est = np.clip(volatility_est, a_min=1e-5, a_max=None)

    # Bước 3: Tạo trọng số W = 1 / Var = 1 / (volatility_est^2)
    weights_tr = 1.0 / (volatility_est**2)

    # Bước 4: Fit WLS sử dụng nhân vector thay vì sinh ma trận đường chéo khổng lồ
    XTw_X = (X_tr * weights_tr[:, None]).T.dot(X_tr)
    XTw_y = (X_tr * weights_tr[:, None]).T.dot(y_tr)
    w_wls = np.linalg.pinv(XTw_X).dot(XTw_y)

    return X_val.dot(w_wls)


evaluate_model(
    "WLS", fit_pred_wls, X_train_b, y_train, X_test_b, y_test, X_val_b, y_val
)

# %% [markdown]
# ## 3. Hồi quy Regularization

# %% [markdown]
# ### 3.1 Ridge Regression (L2)


# %%
def ridge_closed_form(X, y, lam):
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0  # Không regularize bias
    return np.linalg.pinv(X.T @ X + lam * I) @ X.T @ y


def ridge_cv(X, y, lambdas):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_errors = []

    for lam in lambdas:
        fold_errs = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            w = ridge_closed_form(X_tr, y_tr, lam)
            y_pred = X_val @ w
            fold_errs.append(np.mean((y_val - y_pred) ** 2))

        cv_errors.append(np.mean(fold_errs))

    best_idx = np.argmin(cv_errors)
    return lambdas[best_idx], cv_errors


# Grid search
lambdas_ridge = np.logspace(-2, 4, 30)
best_lam_ridge, cv_errs_ridge = ridge_cv(X_train_scaled_b, y_train, lambdas_ridge)

print(f"Optimal Lambda (Ridge): {best_lam_ridge:.4f}")

# Plot CV curve
plt.figure(figsize=(10, 5))
plt.plot(np.log10(lambdas_ridge), cv_errs_ridge, marker="o")
plt.xlabel("log10(Lambda)")
plt.ylabel("CV MSE")
plt.title("Ridge: Grid Search")
plt.show()


# %% [markdown]
# #### Regularization Path (Ridge)

# %%
weights_ridge = []
for lam in lambdas_ridge:
    w = ridge_closed_form(X_train_scaled_b, y_train, lam)
    weights_ridge.append(w[1:])  # bỏ bias

weights_ridge = np.array(weights_ridge)

plt.figure(figsize=(10, 6))
for i in range(weights_ridge.shape[1]):
    plt.plot(np.log10(lambdas_ridge), weights_ridge[:, i])
plt.xlabel("log10(Lambda)")
plt.ylabel("Weights")
plt.title("Ridge Regularization Path")
plt.show()


# %%
def fit_pred_ridge(X_tr, y_tr, X_val):
    w = ridge_closed_form(X_tr, y_tr, best_lam_ridge)
    return X_val @ w


evaluate_model(
    "Ridge (L2)",
    fit_pred_ridge,
    X_train_scaled_b,
    y_train,
    X_test_scaled_b,
    y_test,
)


# %% [markdown]
# ### 3.2 Lasso Regression (L1 - Coordinate Descent)


# %%
def soft_threshold(rho, lam):
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    return 0.0


def lasso_cd(X, y, lam, epochs=30, w_init=None):
    m, n = X.shape
    w = np.zeros(n) if w_init is None else w_init.copy()
    losses = []

    for _ in range(epochs):
        for j in range(n):
            X_j = X[:, j]
            residual = y - (X @ w - w[j] * X_j)
            rho = X_j.T @ residual

            if j == 0:
                w[j] = rho / (X_j.T @ X_j)
            else:
                w[j] = soft_threshold(rho, lam) / (X_j.T @ X_j)

        loss = np.mean((X @ w - y) ** 2) + lam * np.sum(np.abs(w[1:])) / m
        losses.append(loss)

    return w, losses


def lasso_cv(X, y, lambdas, epochs=30):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_errors = []

    warm_starts = [np.zeros(X.shape[1]) for _ in range(10)]

    for lam in lambdas:
        fold_errs = []
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            w, _ = lasso_cd(X_tr, y_tr, lam, epochs, warm_starts[i])
            warm_starts[i] = w

            y_pred = X_val @ w
            fold_errs.append(np.mean((y_val - y_pred) ** 2))

        cv_errors.append(np.mean(fold_errs))

    best_idx = np.argmin(cv_errors)
    return lambdas[best_idx], cv_errors


# Grid search
lambdas_lasso = np.logspace(4, -1, 30)
best_lam_lasso, cv_errs_lasso = lasso_cv(X_train_scaled_b, y_train, lambdas_lasso)

print(f"Optimal Lambda (Lasso): {best_lam_lasso:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(np.log10(lambdas_lasso), cv_errs_lasso, marker="o")
plt.xlabel("log10(Lambda)")
plt.ylabel("CV MSE")
plt.title("Lasso: Grid Search")
plt.show()


# %% [markdown]
# #### Regularization Path (Lasso)

# %%
weights_lasso = []
w_warm = np.zeros(X_train_scaled_b.shape[1])

for lam in lambdas_lasso:
    w_warm, _ = lasso_cd(X_train_scaled_b, y_train, lam, epochs=30, w_init=w_warm)
    weights_lasso.append(w_warm[1:])

weights_lasso = np.array(weights_lasso)

plt.figure(figsize=(10, 6))
for i in range(weights_lasso.shape[1]):
    plt.plot(np.log10(lambdas_lasso), weights_lasso[:, i])

plt.gca().invert_xaxis()
plt.xlabel("log10(Lambda)")
plt.ylabel("Weights")
plt.title("Lasso Regularization Path")
plt.show()


# %%
def fit_pred_lasso(X_tr, y_tr, X_val):
    w, _ = lasso_cd(X_tr, y_tr, best_lam_lasso, epochs=50)
    return X_val @ w


evaluate_model(
    "Lasso (L1)",
    fit_pred_lasso,
    X_train_scaled_b,
    y_train,
    X_test_scaled_b,
    y_test,
)


# %% [markdown]
# ### 3.3 Elastic Net
# Biểu đồ Learning Curves hội tụ bằng Coordinate Descent
# %%
def elastic_net(X, y, lam1=1.0, lam2=1.0, epochs=30):
    m, n = X.shape
    w = np.zeros(n)
    losses = []

    for epoch in range(epochs):
        for j in range(n):
            X_j = X[:, j]
            y_pred_no_j = X.dot(w) - w[j] * X_j
            rho = X_j.T.dot(y - y_pred_no_j)

            if j == 0:
                w[j] = rho / (X_j.T.dot(X_j))
            else:
                w[j] = soft_threshold(rho, m * lam1) / (X_j.T.dot(X_j) + m * lam2)

        # Phạt kép Elastic Net Loss chuẩn: 1/2m * MSE + l1*|w| + l2/2*w^2
        current_loss = (
            np.mean((X.dot(w) - y) ** 2) / 2.0
            + lam1 * np.sum(np.abs(w[1:]))
            + (lam2 / 2.0) * np.sum(w[1:] ** 2)
        )
        losses.append(current_loss)

    return w, losses


lam1_grid = np.logspace(-3, 1, 10)
lam2_grid = np.logspace(-3, 1, 10)
cv_errors_elastic = np.zeros((len(lam1_grid), len(lam2_grid)))

kf_el = KFold(n_splits=5, shuffle=True, random_state=42)
for i, l1 in enumerate(lam1_grid):
    for j, l2 in enumerate(lam2_grid):
        fold_errs = []
        for train_index, val_index in kf_el.split(X_train_scaled_b):
            X_fold_train, X_fold_val = (
                X_train_scaled_b[train_index],
                X_train_scaled_b[val_index],
            )
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            w_opt, _ = elastic_net(
                X_fold_train, y_fold_train, lam1=l1, lam2=l2, epochs=20
            )
            y_pred = X_fold_val.dot(w_opt)
            fold_errs.append(np.mean((y_fold_val - y_pred) ** 2))
        cv_errors_elastic[i, j] = np.mean(fold_errs)

min_idx = np.unravel_index(np.argmin(cv_errors_elastic), cv_errors_elastic.shape)
best_lam1_el = lam1_grid[min_idx[0]]
best_lam2_el = lam2_grid[min_idx[1]]
print(f"Optimal Lam1 for Elastic Net: {best_lam1_el:.5f}")
print(f"Optimal Lam2 for Elastic Net: {best_lam2_el:.5f}")

L1, L2 = np.meshgrid(lam2_grid, lam1_grid)
plt.figure(figsize=(8, 6))
cp = plt.contourf(
    np.log10(L2), np.log10(L1), cv_errors_elastic, levels=20, cmap="viridis"
)
plt.colorbar(cp)
plt.scatter(
    np.log10(best_lam2_el),
    np.log10(best_lam1_el),
    color="red",
    marker="x",
    s=100,
    label="Optimal",
)
plt.xlabel("log10(Lambda 2)")
plt.ylabel("log10(Lambda 1)")
plt.title("Elastic Net 2D Grid Search (CV MSE Contour)")
plt.legend()
plt.show()

w_elastic, elastic_losses = elastic_net(
    X_train_scaled_b, y_train, lam1=best_lam1_el, lam2=best_lam2_el, epochs=50
)

plt.figure(figsize=(10, 6))
plt.plot(
    range(1, 51), elastic_losses, marker="o", color="coral", label="Elastic Net Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Objective Function (MSE + L1 + L2 Penalty)")
plt.title("Elastic Net: Learning Curve (Train Loss vs Epoch)")
plt.legend()
plt.show()


def fit_pred_elastic(X_tr, y_tr, X_val):
    w, _ = elastic_net(X_tr, y_tr, lam1=best_lam1_el, lam2=best_lam2_el, epochs=50)
    return X_val.dot(w)


evaluate_model(
    "Elastic Net",
    fit_pred_elastic,
    X_train_scaled_b,
    y_train,
    X_test_scaled_b,
    y_test,
    X_val_scaled_b,
    y_val,
)

# %% [markdown]
# ### 3.4 Lựa chọn đặc trưng (Feature Selection)
# 1. Lasso Nonzero Coefficients: Các biến có hệ số w khác 0
# 2. Forward Stepwise Selection: Thêm từng biến làm giảm MSE validation nhiều nhất
# 3. Backward Elimination: Chặt dần từ full model biến ít ảnh hưởng nhất
# %%

feature_names = list(
    train_df.drop(columns=features_to_drop + ["cnt"], errors="ignore").columns
)

# 1. Lasso Nonzero
lasso_selected = []
for i, w_val in enumerate(w_lasso[1:]):
    if abs(w_val) > 1e-5:
        lasso_selected.append(feature_names[i])


# Helper measure MSE of a subset
def cv_err_subset(subset_indices):
    if len(subset_indices) == 0:
        return np.mean((y_train - np.mean(y_train)) ** 2)

    kf_sub = KFold(n_splits=5, shuffle=True, random_state=42)
    X_sub = X_train_scaled[:, subset_indices]
    X_sub_b = np.c_[np.ones((len(X_sub), 1)), X_sub]

    cv_errs = []
    for tr_idx, val_idx in kf_sub.split(X_sub_b):
        X_tr_f, X_val_f = X_sub_b[tr_idx], X_sub_b[val_idx]
        y_tr_f, y_val_f = y_train[tr_idx], y_train[val_idx]
        w_f = np.linalg.pinv(X_tr_f.T.dot(X_tr_f)).dot(X_tr_f.T).dot(y_tr_f)
        y_pred_f = X_val_f.dot(w_f)
        cv_errs.append(np.mean((y_val_f - y_pred_f) ** 2))
    return np.mean(cv_errs)


# 2. Forward Stepwise
forward_selected_indices = []
remaining_indices = list(range(len(feature_names)))
best_forward_mse = cv_err_subset([])

while remaining_indices:
    best_candidate = None
    best_candidate_mse = best_forward_mse

    for idx in remaining_indices:
        test_subset = forward_selected_indices + [idx]
        mse = cv_err_subset(test_subset)
        if mse < best_candidate_mse:
            best_candidate = idx
            best_candidate_mse = mse

    if best_candidate is not None:
        forward_selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)
        best_forward_mse = best_candidate_mse
    else:
        break

forward_selected = [feature_names[i] for i in forward_selected_indices]

# 3. Backward Elimination
backward_selected_indices = list(range(len(feature_names)))
best_backward_mse = cv_err_subset(backward_selected_indices)

while len(backward_selected_indices) > 1:
    worst_candidate = None
    best_candidate_mse = best_backward_mse

    for idx in backward_selected_indices:
        test_subset = backward_selected_indices.copy()
        test_subset.remove(idx)
        mse = cv_err_subset(test_subset)
        if mse < best_candidate_mse:
            worst_candidate = idx
            best_candidate_mse = mse

    if worst_candidate is not None:
        backward_selected_indices.remove(worst_candidate)
        best_backward_mse = best_candidate_mse
    else:
        break

backward_selected = [feature_names[i] for i in backward_selected_indices]

print("\n" + "=" * 60)
print("=== TỔNG KẾT FEATURE SELECTION ===")
print("=" * 60)
print(f"Lasso Selected ({len(lasso_selected)} features): {lasso_selected}")
print(
    f"Forward Stepwise Selected ({len(forward_selected)} features): {forward_selected}"
)
print(
    f"Backward Elimination Selected ({len(backward_selected)} features): {backward_selected}"
)

# %% [markdown]
# ## 4. Hàm cơ sở phi tuyến (Sklearn)
# %%
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler

# 1. Polynomial
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_val_poly = poly.transform(X_val_scaled)
X_test_poly = poly.transform(X_test_scaled)


def fit_pred_sk_poly(X_tr, y_tr, X_val):
    model = Ridge(alpha=best_lam_ridge)
    model.fit(X_tr, y_tr)
    return model.predict(X_val)


evaluate_model(
    "Ridge + Polynomial (deg=2)",
    fit_pred_sk_poly,
    X_train_poly,
    y_train,
    X_test_poly,
    y_test,
)

# 2. Gaussian RBF Features
rbf = RBFSampler(gamma=0.1, random_state=42, n_components=100)
X_train_rbf = rbf.fit_transform(X_train_scaled)
X_val_rbf = rbf.transform(X_val_scaled)
X_test_rbf = rbf.transform(X_test_scaled)

evaluate_model(
    "Ridge + Gaussian RBF kernel",
    fit_pred_sk_poly,
    X_train_rbf,
    y_train,
    X_test_rbf,
    y_test,
    X_val_rbf,
    y_val,
)

# %% [markdown]
# ## 5. Mô hình Nâng cao
# %%
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def fit_pred_bayes(X_tr, y_tr, X_val):
    model = BayesianRidge()
    model.fit(X_tr, y_tr)
    return model.predict(X_val)


evaluate_model(
    "Bayesian Ridge",
    fit_pred_bayes,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    X_val_scaled,
    y_val,
)

# %% [markdown]
# #### Khoảng tự tin (Predictive Distribution) của Hồi quy Bayesian
# %%
model_bayes = BayesianRidge()
model_bayes.fit(X_train_scaled, y_train)
y_pred_bayes, y_std_bayes = model_bayes.predict(X_test_scaled, return_std=True)

# Trực quan hoá 100 điểm test ngẫu nhiên để thấy rõ dải tự tin
np.random.seed(42)
indices = np.random.choice(len(y_test), 100, replace=False)
sorted_idx = np.argsort(y_pred_bayes[indices])
idx_to_plot = indices[sorted_idx]

plt.figure(figsize=(12, 6))
plt.plot(y_pred_bayes[idx_to_plot], color="teal", label="Dự đoán (Predicted)", lw=2)
plt.scatter(
    range(100), y_test[idx_to_plot], color="coral", label="Thực tế (Actual)", zorder=3
)
plt.fill_between(
    range(100),
    y_pred_bayes[idx_to_plot] - 2 * y_std_bayes[idx_to_plot],
    y_pred_bayes[idx_to_plot] + 2 * y_std_bayes[idx_to_plot],
    color="lightblue",
    alpha=0.5,
    label="Vùng bất định ±2σ",
)
plt.title("Bayesian Ridge: Predictive Distribution (100 Test Samples)")
plt.ylabel("cnt (Số lượng xe)")
plt.xlabel("Chỉ mục Mẫu (Sắp xếp theo giá trị Dự đoán)")
plt.legend()
plt.show()


def fit_pred_kr(X_tr, y_tr, X_val):
    kr = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1)
    kr.fit(X_tr, y_tr)
    return kr.predict(X_val)


evaluate_model(
    "Kernel Ridge (RBF)", fit_pred_kr, X_train_scaled, y_train, X_test_scaled, y_test
)


def fit_pred_gpr(X_tr, y_tr, X_val):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
    )
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=0.1)
    gpr.fit(X_tr, y_tr)
    return gpr.predict(X_val)


evaluate_model(
    "Gaussian Process",
    fit_pred_gpr,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    X_val_scaled,
    y_val,
)

# %% [markdown]
# #### Khoảng tự tin (Predictive Posterior) của Gaussian Process
# %%
gpr_plot = GaussianProcessRegressor(
    kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
    n_restarts_optimizer=0,
    alpha=0.1,
)
gpr_plot.fit(X_train_scaled, y_train)
y_pred_gp, y_std_gp = gpr_plot.predict(X_test_scaled, return_std=True)

indices_gp = np.random.choice(len(y_test), 50, replace=False)
sorted_idx_gp = np.argsort(y_pred_gp[indices_gp])
idx_to_plot_gp = indices_gp[sorted_idx_gp]

plt.figure(figsize=(12, 6))
plt.plot(y_pred_gp[idx_to_plot_gp], color="teal", label="Dự đoán (Predicted)", lw=2)
plt.scatter(
    range(50), y_test[idx_to_plot_gp], color="coral", label="Thực tế (Actual)", zorder=3
)
plt.fill_between(
    range(50),
    y_pred_gp[idx_to_plot_gp] - 2 * y_std_gp[idx_to_plot_gp],
    y_pred_gp[idx_to_plot_gp] + 2 * y_std_gp[idx_to_plot_gp],
    color="lightgreen",
    alpha=0.5,
    label="Vùng bất định ±2σ",
)
plt.title("Gaussian Process: Posterior Predictive (50 Test Samples)")
plt.ylabel("cnt (Số lượng xe)")
plt.xlabel("Chỉ mục Mẫu (Sắp xếp theo giá trị Dự đoán)")
plt.legend()
plt.show()

# %% [markdown]
# ## 6. Đánh giá Tổng hợp
# ### Learning Curves Mẫu Train theo Data Size
# %%
# Gọi hàm helper đã viết ở phía trên để vẽ đường học tập theo số lượng mẫu
plot_learning_curve_data_size(
    "Ridge Regression", fit_pred_ridge, X_train_scaled_b, y_train, X_val_scaled_b, y_val
)

# Bạn có thể vẽ thêm cho các mô hình khác (Ví dụ OLS) một cách dễ dàng:
plot_learning_curve_data_size(
    "Normal Equations OLS", fit_pred_ols, X_train_b, y_train, X_val_b, y_val
)

# %% [markdown]
# ### Bảng Phân Tích K-Fold & Kiểm định Thống kê
# Danh sách thông số đo Mean/Std Error qua 10-Fold CV. Kiểm định Wilcoxon để xác nhận sự cải thiện giữa các mô hình.
# %%
print("\n" + "=" * 60)
print("BẢNG TỔNG HỢP SO SÁNH TẤT CẢ CÁC MÔ HÌNH")
print("=" * 60)

results_df = pd.DataFrame(metrics_list)
try:
    from IPython.display import display

    display(results_df)  # Hiển thị đẹp nếu chạy trong Jupyter Notebook
except ImportError:
    print(results_df.to_string(index=False))

plt.figure(figsize=(12, 6))
sns.barplot(
    data=results_df.sort_values(by="Test R2", ascending=False),
    x="Test R2",
    y="Model",
    palette="viridis",
)
plt.title("So sánh Tổng kết R2 của các mô hình Hồi quy (cao nhất là tốt nhất)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Biểu đồ Phần dư (Residuals Plot) cho toàn bộ các mô hình
# Theo dõi trực quan phần dư để kiểm tra giả định ngẫu nhiên / đồng nhất phương sai.
# %%
num_models = len(metrics_list)
cols = 3
rows = (num_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()

for i, m in enumerate(metrics_list):
    m_name = m["Model"]
    ax = axes[i]
    m_preds = preds_dict[m_name]
    m_resids = residuals_dict[m_name]

    ax.scatter(m_preds, m_resids, alpha=0.3, color="coral")
    ax.axhline(y=0, color="r", linestyle="--", lw=2)
    ax.set_title(f"{m_name}", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")

for j in range(len(metrics_list), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(
    "Residuals (Thực tế - Dự đoán) vs Predicted - Toàn bộ mô hình", fontsize=16, y=1.02
)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("KIỂM ĐỊNH THỐNG KÊ PAIRED T-TEST HOẶC WILCOXON")
print(
    "Mục đích: Xác nhận xem các mô hình có thực sự khác biệt có ý nghĩa thống kê không."
)
print("=" * 60)

from scipy.stats import ttest_rel


def verify_and_compare_kfold(model_base, model_advanced):
    """
    Sử dụng Paired t-test trên 10 quan sát tương ứng từ 10-Fold CV (CV RMSE).
    """
    rmse_base = cv_scores_dict[model_base]
    rmse_adv = cv_scores_dict[model_advanced]

    t_stat, p_value = ttest_rel(rmse_base, rmse_adv)

    print(
        f"\nKiểm định Paired T-Test (trên K-Fold CV) giữa '{model_base}' và '{model_advanced}':"
    )
    print(f"T-Statistic = {t_stat:.4f}")
    if np.isnan(p_value):
        print("P-Value     = N/A (có thể các Fold RMSE hoàn toàn giống hệt nhau)")
    else:
        print(f"P-Value     = {p_value:.4e}")

        if p_value < 0.05:
            print(
                f"=> Kết luận: P-value < 0.05. Sự khác biệt giữa {model_base} và {model_advanced} là CÓ Ý NGHĨA THỐNG KÊ."
            )
            if np.mean(rmse_adv) < np.mean(rmse_base):
                print(
                    f"=> '{model_advanced}' thực sự hoạt động TỐT HƠN '{model_base}' (RMSE nhỏ hơn)."
                )
            else:
                print(
                    f"=> '{model_base}' thực sự hoạt động TỐT HƠN '{model_advanced}' (RMSE nhỏ hơn)."
                )
        else:
            print(
                f"=> Kết luận: P-value >= 0.05. Không có sự khác biệt có ý nghĩa thống kê."
            )


# Áp dụng Paired T-test CV K-Fold đối chiếu tất cả các mô hình với Baseline OLS
base_model = "Normal Equations OLS"
print(
    f"--- SO SÁNH TOÀN BỘ CÁC MÔ HÌNH VỚI BASELINE ({base_model}) DỰA TRÊN 10-FOLD CV ---"
)

for m in metrics_list:
    current_model = m["Model"]
    if current_model != base_model:
        verify_and_compare_kfold(base_model, current_model)
