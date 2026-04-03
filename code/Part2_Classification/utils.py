import numpy as np
import copy
from scipy import stats

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
