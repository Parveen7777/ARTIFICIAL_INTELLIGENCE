"""
mlp.py  -  Multi-layer perceptron with back-propagation, written from scratch in NumPy.

Notation (matches the report, Task 4.1):
    a^(0) = x                                    input,                 shape (m, n_0)  for a mini-batch of m examples
    z^(l) = a^(l-1) W^(l)T + b^(l)               pre-activation,        shape (m, n_l)
    a^(l) = g_l(z^(l))                           activation,            shape (m, n_l)
    W^(l) has shape (n_l, n_{l-1}),  b^(l) has shape (n_l,)
    delta^(l) = dLoss / dz^(l)                   error term,            shape (m, n_l)

Two output/loss pairs are supported:
    "sigmoid_mse"  : a^(L)=sigmoid(z^(L)),  Loss = mean_over_batch 0.5*||a^(L)-y||^2     (desk-check network)
    "softmax_ce"   : a^(L)=softmax(z^(L)),  Loss = mean_over_batch  -sum y log a^(L)       (digits classifier)
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


ACT = {
    "sigmoid": (sigmoid, lambda z, a: a * (1.0 - a)),
    "relu":    (lambda z: np.maximum(0.0, z), lambda z, a: (z > 0).astype(float)),
}


class MLP:
    def __init__(self, sizes, hidden_act="sigmoid", loss="sigmoid_mse", seed=0):
        self.sizes, self.hidden_act, self.loss_name = list(sizes), hidden_act, loss
        rng = np.random.RandomState(seed)
        self.W, self.b = [None], [None]                         # index 0 unused so that W[l] is layer l
        for l in range(1, len(sizes)):
            fan_in, fan_out = sizes[l - 1], sizes[l]
            if hidden_act == "relu" and l < len(sizes) - 1:     # He initialisation
                w = rng.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
            else:                                               # Xavier / Glorot
                lim = np.sqrt(6.0 / (fan_in + fan_out))
                w = rng.uniform(-lim, lim, size=(fan_out, fan_in))
            self.W.append(w)
            self.b.append(np.zeros(fan_out))
        self.L = len(sizes) - 1

    # ---------------------------------------------------------------- forward recurrence
    def forward(self, X):
        A, Z = [X], [None]
        for l in range(1, self.L + 1):
            z = A[-1] @ self.W[l].T + self.b[l]
            if l == self.L:
                a = sigmoid(z) if self.loss_name == "sigmoid_mse" else softmax(z)
            else:
                a = ACT[self.hidden_act][0](z)
            Z.append(z)
            A.append(a)
        return A, Z

    def loss(self, X, Y):
        A, _ = self.forward(X)
        return self._loss_from_output(A[-1], Y)

    def _loss_from_output(self, out, Y):
        if self.loss_name == "sigmoid_mse":
            return float(np.mean(0.5 * np.sum((out - Y) ** 2, axis=1)))
        return float(-np.mean(np.sum(Y * np.log(np.maximum(out, 1e-12)), axis=1)))

    # ---------------------------------------------------------------- backward recurrence
    def backward(self, X, Y):
        """Return loss, dW (list), db (list), deltas (list) for a mini-batch."""
        m = X.shape[0]
        A, Z = self.forward(X)
        L = self.L
        loss = self._loss_from_output(A[L], Y)
        delta = [None] * (L + 1)
        # boundary condition at the output layer
        if self.loss_name == "sigmoid_mse":
            delta[L] = (A[L] - Y) * A[L] * (1.0 - A[L]) / m
        else:
            delta[L] = (A[L] - Y) / m
        # recurrence from layer L-1 down to 1:  delta^(l) = (delta^(l+1) W^(l+1)) * g'(z^(l))
        for l in range(L - 1, 0, -1):
            delta[l] = (delta[l + 1] @ self.W[l + 1]) * ACT[self.hidden_act][1](Z[l], A[l])
        dW = [None] + [delta[l].T @ A[l - 1] for l in range(1, L + 1)]
        db = [None] + [delta[l].sum(axis=0) for l in range(1, L + 1)]
        return loss, dW, db, delta, A, Z

    def step(self, X, Y, lr):
        loss, dW, db, *_ = self.backward(X, Y)
        for l in range(1, self.L + 1):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]
        return loss

    def grad_norms(self, X, Y):
        """Frobenius norm of dLoss/dW^(l) for every layer (used to diagnose vanishing/exploding gradients)."""
        _, dW, *_ = self.backward(X, Y)
        return [float(np.linalg.norm(dW[l])) for l in range(1, self.L + 1)]

    def predict(self, X):
        A, _ = self.forward(X)
        return A[-1].argmax(axis=1) if A[-1].shape[1] > 1 else (A[-1][:, 0] > 0.5).astype(int)

    # ---------------------------------------------------------------- training loop (mini-batch SGD)
    def fit(self, Xtr, Ytr, lr, epochs, batch_size, seed=0, Xva=None, Yva=None, track_grads=False, stop_loss=None):
        rng = np.random.RandomState(seed)
        hist = dict(train_loss=[], val_loss=[], grad_norms=[])
        n = Xtr.shape[0]
        for ep in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                if track_grads and ep == 0 and s == 0:
                    hist["grad_first_batch"] = self.grad_norms(Xtr[idx], Ytr[idx])
                self.step(Xtr[idx], Ytr[idx], lr)
            tl = self.loss(Xtr, Ytr)
            hist["train_loss"].append(tl)
            if Xva is not None:
                hist["val_loss"].append(self.loss(Xva, Yva))
            if track_grads:
                hist["grad_norms"].append(self.grad_norms(Xtr, Ytr))
            if not np.isfinite(tl):
                break                                   # diverged: stop (stopping criterion)
            if stop_loss is not None and tl < stop_loss:
                break
        return hist

    # ---------------------------------------------------------------- numerical gradient (for verification)
    def numerical_gradients(self, X, Y, eps=1e-5):
        dW = [None] + [np.zeros_like(self.W[l]) for l in range(1, self.L + 1)]
        db = [None] + [np.zeros_like(self.b[l]) for l in range(1, self.L + 1)]
        for l in range(1, self.L + 1):
            for arr, g in ((self.W[l], dW[l]), (self.b[l], db[l])):
                it = np.nditer(arr, flags=["multi_index"])
                for _ in it:
                    ix = it.multi_index
                    old = arr[ix]
                    arr[ix] = old + eps
                    lp = self.loss(X, Y)
                    arr[ix] = old - eps
                    lm = self.loss(X, Y)
                    arr[ix] = old
                    g[ix] = (lp - lm) / (2 * eps)
        return dW, db


class Perceptron:
    """Single-layer perceptron with the classic perceptron learning rule (threshold unit)."""
    def __init__(self, n_in, seed=0):
        rng = np.random.RandomState(seed)
        self.w = rng.randn(n_in) * 0.1
        self.b = 0.0

    def predict(self, X):
        return (X @ self.w + self.b > 0).astype(int)

    def fit(self, X, y, lr=0.1, epochs=1000):
        best = 0.0
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = int(xi @ self.w + self.b > 0)
                upd = lr * (yi - pred)
                self.w += upd * xi
                self.b += upd
                errors += int(upd != 0)
            best = max(best, (self.predict(X) == y).mean())
            if errors == 0:
                break
        return best
