import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def confusion_matrix_df(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    cm = pd.crosstab(
        pd.Categorical(y_true, categories=labels),
        pd.Categorical(y_pred, categories=labels),
        rownames=['True'], colnames=['Pred']
    )
    cm.index = [class_names[i] for i in cm.index]
    cm.columns = [class_names[i] for i in cm.columns]
    return cm

def add_bias(X):
    """ Adds a column of ones to the matrix X for the bias term """
    return np.hstack([np.ones((X.shape[0], 1)), X])

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def softmax(Z):
    """ Z shape: (N, K) """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def make_perceptron_demo_data(n_samples=200, kind='separable', random_state=42):
    """
    Create simple 2D binary datasets to illustrate Perceptron convergence.

    kind='separable'      -> two Gaussian blobs with a linear gap
    kind='non_separable'  -> XOR-style dataset that is not linearly separable
    """
    rng = np.random.default_rng(random_state)
    half = n_samples // 2

    if kind == 'separable':
        X_pos = rng.normal(loc=[2.0, 2.0], scale=0.6, size=(half, 2))
        X_neg = rng.normal(loc=[-2.0, -2.0], scale=0.6, size=(n_samples - half, 2))
        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * (n_samples - half) + [1] * half)
    elif kind == 'non_separable':
        quarter = n_samples // 4
        centers = np.array([
            [-1.5, -1.5],
            [-1.5,  1.5],
            [ 1.5, -1.5],
            [ 1.5,  1.5],
        ])
        blocks = [rng.normal(loc=center, scale=0.45, size=(quarter, 2)) for center in centers]
        X = np.vstack(blocks)
        y = np.array([0] * quarter + [1] * quarter + [1] * quarter + [0] * quarter)
        if X.shape[0] < n_samples:
            extra = n_samples - X.shape[0]
            extra_X = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(extra, 2))
            extra_y = rng.integers(0, 2, size=extra)
            X = np.vstack([X, extra_X])
            y = np.concatenate([y, extra_y])
    else:
        raise ValueError("kind must be either 'separable' or 'non_separable'")

    order = rng.permutation(len(y))
    return X[order], y[order]

def stratified_kfold_indices(y, k=5, shuffle=True, random_state=42):
    """Return stratified train/validation indices for binary or multiclass labels."""
    y = np.asarray(y)
    if k < 2:
        raise ValueError("k must be at least 2")

    rng = np.random.default_rng(random_state)
    unique_classes = np.unique(y)
    fold_buckets = [[] for _ in range(k)]

    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        if shuffle:
            cls_idx = rng.permutation(cls_idx)
        for pos, idx in enumerate(cls_idx):
            fold_buckets[pos % k].append(int(idx))

    folds = []
    all_indices = np.arange(len(y))
    for val_bucket in fold_buckets:
        val_idx = np.array(sorted(val_bucket), dtype=int)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        folds.append((train_idx, val_idx))
    return folds

def compute_weight_sparsity(weights, threshold=1e-3):
    """Measure how many non-bias weights are effectively zero."""
    weights = np.asarray(weights)
    if weights.ndim == 0:
        raise ValueError("weights must be an array")

    core = weights[1:] if weights.ndim == 1 else weights[1:, ...]
    flat = np.ravel(core)
    near_zero = np.abs(flat) <= threshold
    return {
        'threshold': float(threshold),
        'near_zero_count': int(np.sum(near_zero)),
        'total_count': int(flat.size),
        'near_zero_ratio': float(np.mean(near_zero)) if flat.size > 0 else 0.0,
    }

def tune_logistic_reg_stratified_cv(
    X,
    y,
    C_values,
    penalty='l2',
    learning_rate=0.1,
    max_epochs=500,
    class_weight=None,
    k=5,
    random_state=42,
):
    """
    Tune inverse regularization strength C using stratified k-fold CV.
    Returns best C and fold scores for each candidate.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    folds = stratified_kfold_indices(y, k=k, shuffle=True, random_state=random_state)
    summary_rows = []
    best = None

    for C in C_values:
        fold_scores = []
        for train_idx, val_idx in folds:
            model = LogisticRegressionReg(
                learning_rate=learning_rate,
                penalty=penalty,
                C=C,
                max_epochs=max_epochs,
                class_weight=class_weight,
            )
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            fold_scores.append(accuracy(y[val_idx], preds))

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        row = {
            'penalty': penalty,
            'C': float(C),
            'lambda': float(1.0 / C) if C > 0 else 0.0,
            'mean_accuracy': mean_score,
            'std_accuracy': std_score,
        }
        for i, score in enumerate(fold_scores, start=1):
            row[f'fold_{i}'] = float(score)
        summary_rows.append(row)

        is_better = best is None or mean_score > best['mean_accuracy'] + 1e-12
        same_score = best is not None and abs(mean_score - best['mean_accuracy']) <= 1e-12
        if is_better or (same_score and C < best['C']):
            best = row

    return {
        'best_C': best['C'],
        'best_lambda': best['lambda'],
        'best_mean_accuracy': best['mean_accuracy'],
        'results_df': pd.DataFrame(summary_rows),
    }

class BaseClassifier:
    def __init__(self):
        self.w = None
        self.classes_ = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

class LogisticRegressionGD(BaseClassifier):
    def __init__(self, learning_rate=0.1, max_epochs=1000, multi_class='ovr'):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.multi_class = multi_class
        self.loss_history = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        N, D = X.shape
        X_b = add_bias(X)

        if self.multi_class == 'multinomial':
            self.w = np.zeros((D + 1, K))
            # One hot encode
            Y_onehot = np.zeros((N, K))
            for k in range(K):
                Y_onehot[:, k] = (y == self.classes_[k]).astype(int)

            for epoch in range(self.max_epochs):
                Z = X_b @ self.w
                Y_pred = softmax(Z)
                grad = X_b.T @ (Y_pred - Y_onehot) / N
                self.w -= self.learning_rate * grad
                loss = -np.mean(np.sum(Y_onehot * np.log(Y_pred + 1e-9), axis=1))
                self.loss_history.append(loss)
        else:
            if K > 2:
                raise ValueError("Class support limit exceeded for 'ovr' mode. Please use wrapper or multi_class='multinomial'")
            y_bin = (y == self.classes_[1]).astype(int)
            self.w = np.zeros(D + 1)

            for epoch in range(self.max_epochs):
                Y_pred = sigmoid(X_b @ self.w)
                grad = X_b.T @ (Y_pred - y_bin) / N
                self.w -= self.learning_rate * grad
                loss = -np.mean(y_bin * np.log(Y_pred + 1e-9) + (1 - y_bin) * np.log(1 - Y_pred + 1e-9))
                self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        if self.multi_class == 'multinomial':
            return softmax(X_b @ self.w)
        else:
            prob1 = sigmoid(X_b @ self.w)
            prob0 = 1 - prob1
            return np.vstack([prob0, prob1]).T

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class LogisticRegressionNewton(BaseClassifier):
    def __init__(self, max_epochs=100, tol=1e-5):
        super().__init__()
        self.max_epochs = max_epochs
        self.tol = tol
        self.loss_history = []
        self.wall_clock_time = []
        import time
        self.time_func = time.time

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("Newton-Raphson is implemented for binary classification only.")
            
        y_bin = (y == self.classes_[1]).astype(int)
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        start_time = self.time_func()
        
        for epoch in range(self.max_epochs):
            Y_pred = sigmoid(X_b @ self.w)
            grad = X_b.T @ (Y_pred - y_bin) / N
            R_diag = Y_pred * (1 - Y_pred)
            H = (X_b.T * R_diag) @ X_b / N
            H += np.eye(D + 1) * 1e-5
            
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)
                
            delta_w = H_inv @ grad
            self.w -= delta_w
            
            loss = -np.mean(y_bin * np.log(Y_pred + 1e-9) + (1 - y_bin) * np.log(1 - Y_pred + 1e-9))
            self.loss_history.append(loss)
            self.wall_clock_time.append(self.time_func() - start_time)
            
            if np.linalg.norm(delta_w) < self.tol:
                break
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        prob1 = sigmoid(X_b @ self.w)
        prob0 = 1 - prob1
        return np.vstack([prob0, prob1]).T

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class Perceptron(BaseClassifier):
    def __init__(self, learning_rate=1.0, max_epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.errors_history = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("Perceptron is implemented for binary classification.")
            
        y_bin = np.where(y == self.classes_[1], 1, -1)
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        for epoch in range(self.max_epochs):
            misclassified = 0
            for i in range(N):
                y_pred = np.sign(X_b[i] @ self.w)
                if y_pred == 0:
                    y_pred = -1
                if y_pred != y_bin[i]:
                    self.w += self.learning_rate * y_bin[i] * X_b[i]
                    misclassified += 1
            
            self.errors_history.append(misclassified)
            if misclassified == 0:
                break
        return self

    def predict(self, X):
        X_b = add_bias(X)
        preds = np.sign(X_b @ self.w)
        preds[preds == 0] = -1
        return np.where(preds == 1, self.classes_[1], self.classes_[0])


class LogisticRegressionReg(BaseClassifier):
    """ LR with specific penalty (l1/l2) and class weights """
    def __init__(self, learning_rate=0.1, penalty='l2', C=1.0, max_epochs=1000, class_weight=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.lambd = 1.0 / C if C > 0 else 0
        self.max_epochs = max_epochs
        self.class_weight = class_weight
        self.loss_history = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[1]).astype(int)
        
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        if self.class_weight == 'balanced':
            N0 = np.sum(y_bin == 0)
            N1 = np.sum(y_bin == 1)
            w0 = N / (2 * N0) if N0 > 0 else 1
            w1 = N / (2 * N1) if N1 > 0 else 1
            weights = np.where(y_bin == 1, w1, w0)
        else:
            weights = np.ones(N)

        for epoch in range(self.max_epochs):
            Y_pred = sigmoid(X_b @ self.w)
            error = (Y_pred - y_bin) * weights
            grad = X_b.T @ error / N
            
            reg_loss = 0
            if self.penalty == 'l2':
                grad[1:] += self.lambd * self.w[1:] / N
                reg_loss = (self.lambd / (2 * N)) * np.sum(self.w[1:] ** 2)
            elif self.penalty == 'l1':
                grad[1:] += self.lambd * np.sign(self.w[1:]) / N
                reg_loss = (self.lambd / N) * np.sum(np.abs(self.w[1:]))

            self.w -= self.learning_rate * grad
            
            bce = -np.mean(weights * (y_bin * np.log(Y_pred + 1e-9) + (1 - y_bin) * np.log(1 - Y_pred + 1e-9)))
            self.loss_history.append(bce + reg_loss)
            
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        prob1 = sigmoid(X_b @ self.w)
        prob0 = 1 - prob1
        return np.vstack([prob0, prob1]).T

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class OneVsRestClassifier(BaseClassifier):
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.estimators_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            y_c = (y == c).astype(int)
            clf = copy.deepcopy(self.estimator)
            clf.fit(X, y_c)
            self.estimators_.append(clf)
        return self

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], len(self.estimators_)))
        for i, clf in enumerate(self.estimators_):
            if hasattr(clf, "predict_proba"):
                probs[:, i] = clf.predict_proba(X)[:, 1]
            else:
                X_b = add_bias(X)
                raw_s = X_b @ clf.w
                min_v, max_v = raw_s.min(), raw_s.max()
                if max_v - min_v > 0:
                    probs[:, i] = (raw_s - min_v) / (max_v - min_v)
                else:
                    probs[:, i] = raw_s
        sum_probs = np.sum(probs, axis=1, keepdims=True)
        sum_probs[sum_probs == 0] = 1e-9
        return probs / sum_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class OneVsOneClassifier(BaseClassifier):
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.estimators_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        for i in range(K):
            for j in range(i + 1, K):
                c1, c2 = self.classes_[i], self.classes_[j]
                mask = (y == c1) | (y == c2)
                X_p, y_p = X[mask], y[mask]
                
                # Transform to 0 and 1 correctly
                # We map c1 -> 0, c2 -> 1
                y_p_bin = (y_p == c2).astype(int)
                clf = copy.deepcopy(self.estimator)
                clf.fit(X_p, y_p_bin)
                self.estimators_[(c1, c2)] = clf
        return self

    def predict(self, X):
        K = len(self.classes_)
        votes = np.zeros((X.shape[0], K))
        
        for i in range(K):
            for j in range(i + 1, K):
                c1, c2 = self.classes_[i], self.classes_[j]
                clf = self.estimators_[(c1, c2)]
                preds = clf.predict(X)
                
                # Predict returns either 0 or 1 because we trained it with y_p_bin
                # If pred == 0 -> voted for c1 (which is class_[i])
                # If pred == 1 -> voted for c2 (which is class_[j])
                # Wait, if clf is BaseClassifier, predict returns self.classes_[0] or [1] from its own training!
                # Its classes are [0, 1].
                for n, pred in enumerate(preds):
                    if pred == 0:
                        votes[n, i] += 1
                    else:
                        votes[n, j] += 1
                        
        return self.classes_[np.argmax(votes, axis=1)]



class ProbitRegressionGD(BaseClassifier):
    def __init__(self, learning_rate=0.1, max_epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_history = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("Probit is implemented for binary classification.")
            
        y_bin = (y == self.classes_[1]).astype(int)
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        for epoch in range(self.max_epochs):
            Z = X_b @ self.w
            Y_pred = stats.norm.cdf(Z)
            Y_pred = np.clip(Y_pred, 1e-9, 1 - 1e-9)
            pdf_Z = stats.norm.pdf(Z)
            
            error = (Y_pred - y_bin) * pdf_Z / (Y_pred * (1 - Y_pred))
            grad = X_b.T @ error / N
            self.w -= self.learning_rate * grad
            
            loss = -np.mean(y_bin * np.log(Y_pred) + (1 - y_bin) * np.log(1 - Y_pred))
            self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        prob1 = stats.norm.cdf(X_b @ self.w)
        prob0 = 1 - prob1
        return np.vstack([prob0, prob1]).T

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

class LDA_Scratch:
    def __init__(self):
        self.priors = {}
        self.means = {}
        self.cov_matrix = None
        self.cov_inv = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # 1. Tính trung bình và xác suất tiên nghiệm (priors) cho từng lớp
        self.cov_matrix = np.zeros((n_features, n_features))
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            
            # Tính ma trận phân tán nội lớp (Within-class scatter matrix)
            # S_k = (X_c - mu_c)^T (X_c - mu_c)
            diff = X_c - self.means[c]
            self.cov_matrix += np.dot(diff.T, diff)
            
        # 2. Tính ma trận hiệp phương sai chung (Shared Covariance)
        self.cov_matrix /= n_samples
        
        # Thêm nhiễu nhỏ (jitter) vào đường chéo để tránh ma trận suy biến (Singular Matrix)
        self.cov_matrix += np.eye(n_features) * 1e-6
        self.cov_inv = np.linalg.inv(self.cov_matrix)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                # Hàm phân quyết tuyến tính (Discriminant Function cho LDA)
                # delta_k(x) = x^T Sigma^-1 mu_k - 0.5 mu_k^T Sigma^-1 mu_k + ln(P(C_k))
                term1 = np.dot(np.dot(x.T, self.cov_inv), self.means[c])
                term2 = 0.5 * np.dot(np.dot(self.means[c].T, self.cov_inv), self.means[c])
                term3 = np.log(self.priors[c])
                posteriors.append(term1 - term2 + term3)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)


class QDA_Scratch:
    def __init__(self):
        self.priors = {}
        self.means = {}
        self.cov_matrices = {}
        self.cov_invs = {}
        self.cov_dets = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            
            # Tính ma trận hiệp phương sai riêng cho từng lớp (Class-specific Covariance)
            diff = X_c - self.means[c]
            cov_k = np.dot(diff.T, diff) / len(X_c)
            cov_k += np.eye(n_features) * 1e-6 # Chống suy biến
            
            self.cov_matrices[c] = cov_k
            self.cov_invs[c] = np.linalg.inv(cov_k)
            # Sử dụng slogdet để tính định thức an toàn hơn với ma trận nhiều chiều
            _, log_det = np.linalg.slogdet(cov_k)
            self.cov_dets[c] = log_det

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                # Hàm phân quyết toàn phương (Discriminant Function cho QDA)
                # delta_k(x) = -0.5 ln|Sigma_k| - 0.5 (x - mu_k)^T Sigma_k^-1 (x - mu_k) + ln(P(C_k))
                diff = x - self.means[c]
                term1 = -0.5 * self.cov_dets[c]
                term2 = -0.5 * np.dot(np.dot(diff.T, self.cov_invs[c]), diff)
                term3 = np.log(self.priors[c])
                posteriors.append(term1 + term2 + term3)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

def rank_features_fisher(X, y, feature_names):
    classes = np.unique(y)
    n_samples, n_features = X.shape
    global_mean = np.mean(X, axis=0)
    
    fisher_scores = []
    
    for j in range(n_features):
        feature_col = X[:, j]
        
        S_B_j = 0.0 # Between-class variance cho đặc trưng j
        S_W_j = 0.0 # Within-class variance cho đặc trưng j
        
        for c in classes:
            X_c_j = feature_col[y == c]
            n_c = len(X_c_j)
            mean_c_j = np.mean(X_c_j)
            
            S_B_j += n_c * (mean_c_j - global_mean[j]) ** 2
            S_W_j += np.sum((X_c_j - mean_c_j) ** 2)
        
        # Tránh chia cho 0
        score = S_B_j / (S_W_j + 1e-10)
        fisher_scores.append(score)
        
    # Sắp xếp đặc trưng theo độ phân biệt giảm dần
    ranked_indices = np.argsort(fisher_scores)[::-1]
    
    print("Ranking features by Fisher Ratio J(w):")
    for rank, idx in enumerate(ranked_indices):
        print(f"{rank+1}. {feature_names[idx]} (J = {fisher_scores[idx]:.4f})")
    
    return ranked_indices, fisher_scores

def project_lda_2d_and_plot(X, y, class_names):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    global_mean = np.mean(X, axis=0)
    
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))
    
    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0).reshape(-1, 1)
        
        # S_W
        diff = X_c - mean_c.T
        S_W += np.dot(diff.T, diff)
        
        # S_B
        mean_diff = mean_c - global_mean.reshape(-1, 1)
        S_B += n_c * np.dot(mean_diff, mean_diff.T)
        
    # Tính S_W^-1 * S_B
    S_W_inv = np.linalg.inv(S_W + np.eye(n_features) * 1e-6)
    matrix_to_solve = np.dot(S_W_inv, S_B)
    
    # Tìm Eigenvalues và Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix_to_solve)
    
    # Sắp xếp giảm dần
    sorted_indices = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, sorted_indices[:2]]  # Ma trận chiếu (n_features x 2)
    
    # Chiếu dữ liệu xuống 2D
    X_lda_2d = np.dot(X, W)
    
    # ---- VẼ ĐƯỜNG BIÊN QUYẾT ĐỊNH ----
    # Huấn luyện một mô hình LDA mới trên dữ liệu 2D để vẽ biên
    lda_2d = LDA_Scratch()
    lda_2d.fit(X_lda_2d, y)
    
    x_min, x_max = X_lda_2d[:, 0].min() - 1, X_lda_2d[:, 0].max() + 1
    y_min, y_max = X_lda_2d[:, 1].min() - 1, X_lda_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = lda_2d.predict(grid_points)
    
    # Map nhãn (int) sang số 0..K-1 để contourf hiểu
    label_to_idx = {c: i for i, c in enumerate(classes)}
    Z_numeric = np.array([label_to_idx[z] for z in Z])
    Z_numeric = Z_numeric.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap='viridis')
    
    for c in classes:
        idx = (y == c)
        label_name = class_names[c] if class_names is not None and c < len(class_names) else f'Class {c}'
        plt.scatter(X_lda_2d[idx, 0], X_lda_2d[idx, 1],
                    label=label_name, edgecolor='k', alpha=0.8)
        
    plt.title("LDA projection onto 2D and display the decision boundary")
    plt.xlabel("Linear Discriminant 1 (LD1)")
    plt.ylabel("Linear Discriminant 2 (LD2)")
    plt.legend()
    plt.show()


class LaplaceApproximationLR(BaseClassifier):
    def __init__(self, lambd=1.0, max_epochs=100, tol=1e-5):
        super().__init__()
        self.lambd = lambd
        self.max_epochs = max_epochs
        self.tol = tol
        self.S_N = None 

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[1]).astype(int)
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        for epoch in range(self.max_epochs):
            Y_pred = sigmoid(X_b @ self.w)
            grad_prior = self.lambd * self.w / N
            grad_prior[0] = 0 
            grad = (X_b.T @ (Y_pred - y_bin) / N) + grad_prior
            
            R_diag = Y_pred * (1 - Y_pred)
            H = (X_b.T * R_diag) @ X_b / N
            H_prior = np.eye(D + 1) * self.lambd / N
            H_prior[0, 0] = 0
            H += H_prior + np.eye(D + 1) * 1e-5
            
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)
                
            delta_w = H_inv @ grad
            self.w -= delta_w
            if np.linalg.norm(delta_w) < self.tol:
                break
                
        Y_pred = sigmoid(X_b @ self.w)
        R_diag = Y_pred * (1 - Y_pred)
        H_unnorm = (X_b.T * R_diag) @ X_b
        H_prior_unnorm = np.eye(D + 1) * self.lambd
        H_prior_unnorm[0, 0] = 0
        H_unnorm += H_prior_unnorm + np.eye(D + 1) * 1e-5
        
        try:
            self.S_N = np.linalg.inv(H_unnorm)
        except np.linalg.LinAlgError:
            self.S_N = np.linalg.pinv(H_unnorm)
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        mu_a = X_b @ self.w
        var_a = np.sum((X_b @ self.S_N) * X_b, axis=1)
        kappa = 1.0 / np.sqrt(1 + np.pi * var_a / 8)
        prob1 = sigmoid(kappa * mu_a)
        prob0 = 1 - prob1
        return np.vstack([prob0, prob1]).T
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def rbf_kernel(X1, X2, gamma=1.0):
    """Compute the RBF kernel matrix exp(-gamma ||x - x'||^2)."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    x1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    x2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
    sq_dist = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-gamma * sq_dist)


class KernelLogisticRegressionRBF(BaseClassifier):
    """
    Kernel Logistic Regression with RBF kernel.

    Representer form:
        f(x) = sum_i alpha_i K(x_i, x)

    Objective:
        BCE(sigmoid(K alpha), y) + 0.5 * lambda * alpha^T K alpha
    """
    def __init__(self, gamma=1.0, lambd=1e-2, learning_rate=0.1, max_epochs=1000, tol=1e-6):
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.alpha = None
        self.X_train_ = None
        self.K_train_ = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Kernel Logistic Regression is implemented for binary classification only.")

        y_bin = (y == self.classes_[1]).astype(float)
        N = X.shape[0]
        self.X_train_ = X
        self.K_train_ = rbf_kernel(X, X, gamma=self.gamma)
        self.alpha = np.zeros(N, dtype=float)
        self.loss_history = []

        prev_loss = None
        for _ in range(self.max_epochs):
            logits = self.K_train_ @ self.alpha
            probs = sigmoid(logits)
            data_loss = -np.mean(
                y_bin * np.log(probs + 1e-9) + (1.0 - y_bin) * np.log(1.0 - probs + 1e-9)
            )
            reg_loss = 0.5 * self.lambd * (self.alpha @ self.K_train_ @ self.alpha)
            loss = float(data_loss + reg_loss)
            self.loss_history.append(loss)

            grad = (self.K_train_ @ (probs - y_bin)) / N + self.lambd * (self.K_train_ @ self.alpha)
            self.alpha -= self.learning_rate * grad

            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def decision_function(self, X):
        K_test = rbf_kernel(np.asarray(X, dtype=float), self.X_train_, gamma=self.gamma)
        return K_test @ self.alpha

    def predict_proba(self, X):
        logits = self.decision_function(X)
        prob1 = sigmoid(logits)
        prob0 = 1.0 - prob1
        return np.vstack([prob0, prob1]).T

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class GaussianNaiveBayesScratch(BaseClassifier):
    """Gaussian Naive Bayes from scratch for multiclass classification."""
    def __init__(self, var_smoothing=1e-9):
        super().__init__()
        self.var_smoothing = var_smoothing
        self.priors_ = None
        self.means_ = None
        self.vars_ = None
        self.epsilon_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes, dtype=float)
        self.means_ = np.zeros((n_classes, n_features), dtype=float)
        self.vars_ = np.zeros((n_classes, n_features), dtype=float)
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.priors_[idx] = X_c.shape[0] / X.shape[0]
            self.means_[idx] = np.mean(X_c, axis=0)
            self.vars_[idx] = np.var(X_c, axis=0) + self.epsilon_
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        all_log_prob = []
        for idx in range(len(self.classes_)):
            mean = self.means_[idx]
            var = self.vars_[idx]
            log_prior = np.log(self.priors_[idx] + 1e-12)
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            all_log_prob.append(log_prior + log_likelihood)
        return np.vstack(all_log_prob).T

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        shifted = jll - np.max(jll, axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def make_kernel_xor_data(n_samples=240, noise=0.45, random_state=42):
    """Create an XOR-like binary dataset for testing non-linear classifiers."""
    rng = np.random.default_rng(random_state)
    quarter = n_samples // 4
    centers = np.array([
        [-1.5, -1.5],
        [-1.5,  1.5],
        [ 1.5, -1.5],
        [ 1.5,  1.5],
    ])
    X_parts = [rng.normal(loc=center, scale=noise, size=(quarter, 2)) for center in centers]
    X = np.vstack(X_parts)
    y = np.array([0] * quarter + [1] * quarter + [1] * quarter + [0] * quarter)

    if X.shape[0] < n_samples:
        extra = n_samples - X.shape[0]
        extra_X = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(extra, 2))
        extra_y = rng.integers(0, 2, size=extra)
        X = np.vstack([X, extra_X])
        y = np.concatenate([y, extra_y])

    order = rng.permutation(len(y))
    return X[order], y[order]


def train_test_split_custom(X, y, test_size=0.3, shuffle=True, random_state=42):
    """Simple train/test split helper without external dependencies."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    if shuffle:
        indices = rng.permutation(indices)

    n_test = int(np.ceil(n * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def stratified_train_test_split_custom(X, y, test_size=0.3, shuffle=True, random_state=42):
    """Stratified train/test split without external dependencies."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    train_parts_X, test_parts_X = [], []
    train_parts_y, test_parts_y = [], []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if shuffle:
            cls_idx = rng.permutation(cls_idx)

        n_test_cls = max(1, int(np.ceil(len(cls_idx) * test_size)))
        test_idx = cls_idx[:n_test_cls]
        train_idx = cls_idx[n_test_cls:]

        if train_idx.size == 0:
            train_idx = test_idx[:1]
            test_idx = test_idx[1:]

        train_parts_X.append(X[train_idx])
        test_parts_X.append(X[test_idx])
        train_parts_y.append(y[train_idx])
        test_parts_y.append(y[test_idx])

    X_train = np.vstack(train_parts_X)
    X_test = np.vstack(test_parts_X)
    y_train = np.concatenate(train_parts_y)
    y_test = np.concatenate(test_parts_y)

    if shuffle:
        train_order = rng.permutation(len(y_train))
        test_order = rng.permutation(len(y_test))
        X_train, y_train = X_train[train_order], y_train[train_order]
        X_test, y_test = X_test[test_order], y_test[test_order]

    return X_train, X_test, y_train, y_test


def make_gnb_lda_demo_data(
    n_samples_per_class=120,
    case='correlated_shared_cov',
    random_state=42,
):
    """
    Synthetic multiclass Gaussian data for comparing GNB and LDA.

    case='correlated_shared_cov':
        shared correlated covariance -> favors LDA assumptions
    case='independent_small_sample':
        diagonal covariance with fewer train samples -> often favors GNB in practice
    """
    rng = np.random.default_rng(random_state)

    if case == 'correlated_shared_cov':
        means = np.array([
            [-2.0, -2.0,  0.0],
            [ 2.0,  2.0,  0.5],
            [ 0.0,  3.0, -0.5],
        ])
        cov = np.array([
            [1.0, 0.8, 0.5],
            [0.8, 1.2, 0.4],
            [0.5, 0.4, 1.0],
        ])
        covs = [cov, cov, cov]
    elif case == 'independent_small_sample':
        means = np.array([
            [-2.5, 0.0,  2.0, -1.5,  0.5,  1.5],
            [ 2.0, 1.0, -2.0,  1.5, -0.5, -1.0],
            [ 0.0,-2.5,  1.5,  2.0,  1.0, -1.5],
        ])
        diag_0 = np.array([0.6, 0.5, 0.4, 0.7, 0.5, 0.6])
        diag_1 = np.array([0.5, 0.6, 0.5, 0.4, 0.7, 0.5])
        diag_2 = np.array([0.4, 0.7, 0.6, 0.5, 0.6, 0.4])
        covs = [np.diag(diag_0), np.diag(diag_1), np.diag(diag_2)]
    else:
        raise ValueError("Unsupported case for GNB/LDA demo data.")

    X_parts = []
    y_parts = []
    for idx, mean in enumerate(means):
        X_parts.append(rng.multivariate_normal(mean, covs[idx], size=n_samples_per_class))
        y_parts.append(np.full(n_samples_per_class, idx))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    order = rng.permutation(len(y))
    return X[order], y[order]


def linear_classifier_vc_dimension(n_features, include_bias=True):
    """
    VC dimension for linear separators in R^D.

    include_bias=True  -> affine hyperplanes: h = D + 1
    include_bias=False -> homogeneous hyperplanes through origin: h = D
    """
    return int(n_features + 1) if include_bias else int(n_features)


def vc_generalization_bound(empirical_error, vc_dim, n_samples, delta=0.05):
    """
    A standard VC upper bound:
        R(f) <= R_emp(f) + sqrt((h(log(2n/h)+1) + log(4/delta)) / n)
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if vc_dim <= 0:
        raise ValueError("vc_dim must be positive")

    h = min(float(vc_dim), float(max(1, n_samples)))
    complexity = h * (np.log((2.0 * n_samples) / h) + 1.0) + np.log(4.0 / delta)
    margin = np.sqrt(max(complexity, 0.0) / n_samples)
    return {
        'empirical_error': float(empirical_error),
        'vc_dim': int(vc_dim),
        'n_samples': int(n_samples),
        'delta': float(delta),
        'complexity_term': float(complexity),
        'margin': float(margin),
        'bound': float(empirical_error + margin),
    }


def structural_risk_summary(model_specs, n_samples, delta=0.05):
    """
    Build a DataFrame summarizing empirical risk and VC-based complexity.

    Each item in model_specs should contain:
        {
            'model': str,
            'empirical_error': float,
            'vc_dim': int
        }
    """
    rows = []
    for spec in model_specs:
        bound = vc_generalization_bound(
            empirical_error=spec['empirical_error'],
            vc_dim=spec['vc_dim'],
            n_samples=n_samples,
            delta=delta,
        )
        rows.append({
            'model': spec['model'],
            'empirical_error': bound['empirical_error'],
            'vc_dim': bound['vc_dim'],
            'margin': bound['margin'],
            'bound': bound['bound'],
        })
    return pd.DataFrame(rows).sort_values('bound').reset_index(drop=True)


class ProbitRegressionGD(BaseClassifier):
    def __init__(self, learning_rate=0.1, max_epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_history = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("Probit is implemented for binary classification.")
            
        y_bin = (y == self.classes_[1]).astype(int)
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        for epoch in range(self.max_epochs):
            Z = X_b @ self.w
            Y_pred = stats.norm.cdf(Z)
            Y_pred = np.clip(Y_pred, 1e-9, 1 - 1e-9)
            pdf_Z = stats.norm.pdf(Z)
            
            error = (Y_pred - y_bin) * pdf_Z / (Y_pred * (1 - Y_pred))
            grad = X_b.T @ error / N
            self.w -= self.learning_rate * grad
            
            loss = -np.mean(y_bin * np.log(Y_pred) + (1 - y_bin) * np.log(1 - Y_pred))
            self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        prob1 = stats.norm.cdf(X_b @ self.w)
        prob0 = 1 - prob1
        return np.vstack([prob0, prob1]).T

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class LaplaceApproximationLR(BaseClassifier):
    def __init__(self, lambd=1.0, max_epochs=100, tol=1e-5):
        super().__init__()
        self.lambd = lambd
        self.max_epochs = max_epochs
        self.tol = tol
        self.S_N = None 

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[1]).astype(int)
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        for epoch in range(self.max_epochs):
            Y_pred = sigmoid(X_b @ self.w)
            grad_prior = self.lambd * self.w / N
            grad_prior[0] = 0 
            grad = (X_b.T @ (Y_pred - y_bin) / N) + grad_prior
            
            R_diag = Y_pred * (1 - Y_pred)
            H = (X_b.T * R_diag) @ X_b / N
            H_prior = np.eye(D + 1) * self.lambd / N
            H_prior[0, 0] = 0
            H += H_prior + np.eye(D + 1) * 1e-5
            
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)
                
            delta_w = H_inv @ grad
            self.w -= delta_w
            if np.linalg.norm(delta_w) < self.tol:
                break
                
        Y_pred = sigmoid(X_b @ self.w)
        R_diag = Y_pred * (1 - Y_pred)
        H_unnorm = (X_b.T * R_diag) @ X_b
        H_prior_unnorm = np.eye(D + 1) * self.lambd
        H_prior_unnorm[0, 0] = 0
        H_unnorm += H_prior_unnorm + np.eye(D + 1) * 1e-5
        
        try:
            self.S_N = np.linalg.inv(H_unnorm)
        except np.linalg.LinAlgError:
            self.S_N = np.linalg.pinv(H_unnorm)
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        mu_a = X_b @ self.w
        var_a = np.sum((X_b @ self.S_N) * X_b, axis=1)
        kappa = 1.0 / np.sqrt(1 + np.pi * var_a / 8)
        prob1 = sigmoid(kappa * mu_a)
        prob0 = 1 - prob1
        return np.vstack([prob0, prob1]).T
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class ProbitRegressionGD(BaseClassifier):
    def __init__(self, learning_rate=0.1, max_epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_history = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("Probit is implemented for binary classification.")
            
        y_bin = (y == self.classes_[1]).astype(int)
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        for epoch in range(self.max_epochs):
            Z = X_b @ self.w
            Y_pred = stats.norm.cdf(Z)
            Y_pred = np.clip(Y_pred, 1e-9, 1 - 1e-9)
            pdf_Z = stats.norm.pdf(Z)
            
            error = (Y_pred - y_bin) * pdf_Z / (Y_pred * (1 - Y_pred))
            grad = X_b.T @ error / N
            self.w -= self.learning_rate * grad
            
            loss = -np.mean(y_bin * np.log(Y_pred) + (1 - y_bin) * np.log(1 - Y_pred))
            self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        prob1 = stats.norm.cdf(X_b @ self.w)
        prob0 = 1 - prob1
        return np.vstack([prob0, prob1]).T

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class LaplaceApproximationLR(BaseClassifier):
    def __init__(self, lambd=1.0, max_epochs=100, tol=1e-5):
        super().__init__()
        self.lambd = lambd
        self.max_epochs = max_epochs
        self.tol = tol
        self.S_N = None 

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[1]).astype(int)
        N, D = X.shape
        X_b = add_bias(X)
        self.w = np.zeros(D + 1)
        
        for epoch in range(self.max_epochs):
            Y_pred = sigmoid(X_b @ self.w)
            grad_prior = self.lambd * self.w / N
            grad_prior[0] = 0 
            grad = (X_b.T @ (Y_pred - y_bin) / N) + grad_prior
            
            R_diag = Y_pred * (1 - Y_pred)
            H = (X_b.T * R_diag) @ X_b / N
            H_prior = np.eye(D + 1) * self.lambd / N
            H_prior[0, 0] = 0
            H += H_prior + np.eye(D + 1) * 1e-5
            
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)
                
            delta_w = H_inv @ grad
            self.w -= delta_w
            if np.linalg.norm(delta_w) < self.tol:
                break
                
        Y_pred = sigmoid(X_b @ self.w)
        R_diag = Y_pred * (1 - Y_pred)
        H_unnorm = (X_b.T * R_diag) @ X_b
        H_prior_unnorm = np.eye(D + 1) * self.lambd
        H_prior_unnorm[0, 0] = 0
        H_unnorm += H_prior_unnorm + np.eye(D + 1) * 1e-5
        
        try:
            self.S_N = np.linalg.inv(H_unnorm)
        except np.linalg.LinAlgError:
            self.S_N = np.linalg.pinv(H_unnorm)
        return self

    def predict_proba(self, X):
        X_b = add_bias(X)
        mu_a = X_b @ self.w
        var_a = np.sum((X_b @ self.S_N) * X_b, axis=1)
        kappa = 1.0 / np.sqrt(1 + np.pi * var_a / 8)
        prob1 = sigmoid(kappa * mu_a)
        prob0 = 1 - prob1
        return np.vstack([prob0, prob1]).T
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
