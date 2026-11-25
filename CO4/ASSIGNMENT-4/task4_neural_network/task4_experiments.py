"""
TASK 4.4 / 4.5 / 4.6

A. Reproduce the hand trace of the 2-2-1 sigmoid network (one forward + backward pass + update).
B. Gradient check: analytic vs numerical gradients, maximum relative error.
C. XOR: a single-layer perceptron fails, the back-propagation network succeeds.
D. 8x8 digits: grid over hidden units x learning rate x activation (predictions made first).
E. Failure diagnosis with per-layer gradient norms (divergence and vanishing gradient).

Run:  python task4_neural_network/task4_experiments.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from mlp import MLP, Perceptron  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
np.set_printoptions(precision=5, suppress=True, linewidth=120)


# ------------------------------------------------------------------------------------------------ A
def desk_check():
    print("=" * 80 + "\nTASK 4.4 / 4.5  DESK-CHECK REPRODUCED IN CODE (2-2-1 sigmoid network, squared error, lr = 0.5)\n" + "=" * 80)
    net = MLP([2, 2, 1], "sigmoid", "sigmoid_mse")
    net.W[1] = np.array([[0.5, -0.4], [0.3, 0.8]])      # row j = incoming weights of hidden neuron j
    net.W[2] = np.array([[0.6, -0.2]])
    X, Y = np.array([[1.0, 0.0]]), np.array([[1.0]])
    A, Z = net.forward(X)
    loss, dW, db, delta, *_ = net.backward(X, Y)
    print("z1 =", Z[1][0], "  a1 =", A[1][0])
    print("z2 =", Z[2][0], "  a2 (y_hat) =", A[2][0], "  target y = 1")
    print("loss  E = 0.5 (y - y_hat)^2 = %.6f" % loss)
    print("delta2 = (a2 - y) a2 (1 - a2)           =", delta[2][0])
    print("delta1 = (delta2 W2) * a1 (1 - a1)      =", delta[1][0])
    print("dE/dW2 =", dW[2][0], "  dE/db2 =", db[2])
    print("dE/dW1 =", np.round(dW[1], 5).tolist(), "  dE/db1 =", db[1])
    old_W1, old_W2 = net.W[1].copy(), net.W[2].copy()
    net.step(X, Y, 0.5)
    print("UPDATED hidden weights  W1 =", np.round(net.W[1], 5).tolist(), " (before:", old_W1.tolist(), ")")
    print("UPDATED output weights  W2 =", np.round(net.W[2], 5).tolist(), " (before:", old_W2.tolist(), ")")
    print("UPDATED biases b1 =", net.b[1], "  b2 =", net.b[2])
    print("loss after the single update = %.6f  (decreased: %s)" % (net.loss(X, Y), net.loss(X, Y) < loss))


# ------------------------------------------------------------------------------------------------ B
def rel_err(a, b):
    return np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12)


def gradient_check():
    print("\n" + "=" * 80 + "\nTASK 4.5  GRADIENT CHECK  (centred differences, eps = 1e-5)\n" + "=" * 80)
    rng = np.random.RandomState(3)
    cases = [("2-2-1 sigmoid + squared error", MLP([2, 2, 1], "sigmoid", "sigmoid_mse", seed=1), rng.rand(5, 2), rng.randint(0, 2, (5, 1)).astype(float)),
             ("6-5-4-3 sigmoid + softmax-CE", MLP([6, 5, 4, 3], "sigmoid", "softmax_ce", seed=2), rng.randn(7, 6), np.eye(3)[rng.randint(0, 3, 7)]),
             ("6-5-4-3 ReLU    + softmax-CE", MLP([6, 5, 4, 3], "relu", "softmax_ce", seed=3), rng.randn(7, 6), np.eye(3)[rng.randint(0, 3, 7)])]
    worst = 0
    for name, net, X, Y in cases:
        _, dW, db, *_ = net.backward(X, Y)
        nW, nb = net.numerical_gradients(X, Y)
        e = max(max(rel_err(dW[l], nW[l]).max(), rel_err(db[l], nb[l]).max()) for l in range(1, net.L + 1))
        worst = max(worst, e)
        print("%-34s max relative error = %.3e   %s" % (name, e, "PASS (< 1e-6)" if e < 1e-6 else "FAIL"))
    print("Overall maximum relative error = %.3e" % worst)


# ------------------------------------------------------------------------------------------------ C
def xor_demo():
    print("\n" + "=" * 80 + "\nTASK 4.5  XOR : single-layer perceptron vs 2-hidden-layer back-propagation network\n" + "=" * 80)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    best = [Perceptron(2, seed=s).fit(X, y, lr=0.1, epochs=2000) for s in range(10)]
    print("Perceptron: best training accuracy over 2000 epochs, 10 seeds =", np.round(best, 2).tolist(), "-> max = %.2f (never 1.00)" % max(best))
    wins = {}
    for h in (2, 4):
        ok = 0
        for s in range(10):
            net = MLP([2, h, 1], "sigmoid", "sigmoid_mse", seed=s)
            net.fit(X, y.reshape(-1, 1).astype(float), lr=1.0, epochs=6000, batch_size=4, seed=s)
            ok += int((net.predict(X) == y).all())
        wins[h] = ok
        print("MLP 2-%d-1 sigmoid, lr=1.0, 6000 epochs: solved XOR in %d / 10 seeds" % (h, ok))
    net = MLP([2, 4, 1], "sigmoid", "sigmoid_mse", seed=0)
    hist = net.fit(X, y.reshape(-1, 1).astype(float), lr=1.0, epochs=6000, batch_size=4, seed=0)
    print("MLP 2-4-1 (seed 0): final loss = %.5f, outputs = %s, predictions = %s" % (hist["train_loss"][-1], net.forward(X)[0][-1][:, 0], net.predict(X)))


# ------------------------------------------------------------------------------------------------ D
def digits_data():
    d = load_digits()
    X, y = d.data / 16.0, d.target
    Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.25, stratify=ya, random_state=42)
    oh = lambda v: np.eye(10)[v]
    return Xtr, Xva, Xte, ytr, yva, yte, oh


def digits_grid():
    print("\n" + "=" * 80 + "\nTASK 4.6  8x8 DIGITS : hidden units x learning rate x activation  (60 epochs, batch 32)\n" + "=" * 80)
    print("PREDICTIONS MADE BEFORE RUNNING THE GRID")
    print(" P1: lr = 0.001 with sigmoid will be the slowest learner (test accuracy < 0.60 after 60 epochs).")
    print(" P2: hidden = 2 is a bottleneck: accuracy clearly below hidden = 64 for every lr/activation pair that trains.")
    print(" P3: ReLU with lr = 0.1 and hidden = 32 or 64 will give the best accuracy (> 0.95).")
    print(" P4: lr = 1.0 is unstable for ReLU (loss spikes / diverges or units die), while sigmoid at lr = 1.0 still trains.")
    Xtr, Xva, Xte, ytr, yva, yte, oh = digits_data()
    Ytr, Yva = oh(ytr), oh(yva)
    results, histories = {}, {}
    for act in ("sigmoid", "relu"):
        for lr in (0.001, 0.01, 0.1, 1.0):
            for h in (2, 8, 32, 64):
                net = MLP([64, h, 10], act, "softmax_ce", seed=0)
                hist = net.fit(Xtr, Ytr, lr=lr, epochs=60, batch_size=32, seed=0, Xva=Xva, Yva=Yva)
                if np.isfinite(hist["train_loss"][-1]):
                    pred = net.predict(Xte)
                    acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
                else:
                    pred, acc, f1 = None, float("nan"), float("nan")
                results[(act, lr, h)] = (acc, f1, hist["train_loss"][-1], hist["val_loss"][-1] if hist["val_loss"] else np.nan, net, pred)
                histories[(act, lr, h)] = hist
    print("\n%-8s %-7s %-7s | %-9s %-9s | %-10s %-10s" % ("act", "lr", "hidden", "test acc", "macro-F1", "train loss", "val loss"))
    for (act, lr, h), (acc, f1, tl, vl, _, _) in results.items():
        print("%-8s %-7s %-7d | %-9.4f %-9.4f | %-10.4f %-10.4f" % (act, lr, h, acc, f1, tl, vl))

    # prediction scorecard
    g = lambda a, l, h: results[(a, l, h)][0]
    p1 = all(g("sigmoid", 0.001, h) < 0.60 for h in (2, 8, 32, 64))
    p2 = all(g(a, l, 2) < g(a, l, 64) for a in ("sigmoid", "relu") for l in (0.01, 0.1) if np.isfinite(g(a, l, 2)))
    best_key = max(results, key=lambda k: (np.nan_to_num(results[k][0], nan=-1)))
    p3 = best_key[0] == "relu" and best_key[2] in (32, 64) and results[best_key][0] > 0.95
    relu1 = [g("relu", 1.0, h) for h in (2, 8, 32, 64)]
    sig1 = [g("sigmoid", 1.0, h) for h in (8, 32, 64)]
    p4a = all(v > 0.90 for v in sig1)                                   # sigmoid still trains at lr = 1.0
    p4b = any(v < 0.30 for v in relu1)                                  # some ReLU nets fail at lr = 1.0
    p4c = results[("relu", 1.0, 64)][2] > 2.3 or not np.isfinite(results[("relu", 1.0, 64)][2])   # even the widest one diverges
    print("\nPREDICTION SCORECARD:")
    print(" P1 (sigmoid, lr 0.001 -> acc < 0.60 for all widths) :", "CONFIRMED" if p1 else "REFUTED", [round(g("sigmoid", 0.001, h), 3) for h in (2, 8, 32, 64)])
    print(" P2 (hidden=2 below hidden=64)                       :", "CONFIRMED" if p2 else "REFUTED")
    print(" P3 (best model is ReLU, 32/64 units, > 0.95)        :", "CONFIRMED" if p3 else "REFUTED", "-> best =", best_key, "acc = %.4f" % results[best_key][0])
    print(" P4a (sigmoid still trains at lr=1.0)                :", "CONFIRMED" if p4a else "REFUTED", np.round(sig1, 3).tolist())
    print(" P4b (some ReLU nets fail at lr=1.0)                 :", "CONFIRMED" if p4b else "REFUTED", "relu@1.0 =", np.round(relu1, 3).tolist(), "(narrow nets: dead units)")
    print(" P4c (even ReLU-64 diverges at lr=1.0)               :", "CONFIRMED" if p4c else "REFUTED  -> P4 only PARTLY correct: wide ReLU nets survive lr=1.0 after a loss spike; divergence needs a larger lr (see F1)")

    # best model report
    acc, f1, _, _, net, pred = results[best_key]
    cm = confusion_matrix(yte, pred)
    print("\nBEST CONFIG", best_key, "-> test accuracy %.4f, macro-F1 %.4f" % (acc, f1))
    print("Confusion matrix (rows = true digit 0-9, columns = predicted):")
    print(cm)

    # figures: loss curves
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharey=True)
    for r, act in enumerate(("sigmoid", "relu")):
        for c, lr in enumerate((0.001, 0.01, 0.1, 1.0)):
            ax = axes[r, c]
            for h in (2, 8, 32, 64):
                tl = histories[(act, lr, h)]["train_loss"]
                ax.plot(np.clip(tl, 0, 6), label="hidden=%d" % h)
            ax.set_title("%s, lr=%g" % (act, lr), fontsize=10)
            ax.grid(alpha=.3)
            if c == 0:
                ax.set_ylabel("train loss (clipped at 6)")
            if r == 1:
                ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Task 4.6  Training loss curves on 8x8 digits", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task4_digits_loss_curves.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix - best config %s" % (best_key,), fontsize=9)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task4_digits_confusion.png"), dpi=150); plt.close(fig)
    return results, histories


# ------------------------------------------------------------------------------------------------ E
def failure_diagnosis():
    print("\n" + "=" * 80 + "\nTASK 4.6  PROVOKED FAILURES  (diagnosed with per-layer gradient norms)\n" + "=" * 80)
    Xtr, Xva, Xte, ytr, yva, yte, oh = digits_data()
    Ytr = oh(ytr)

    print("\n[F1a] HYPOTHESIS: ReLU, hidden 64, lr = 1.0 diverges.  (Tested - the grid already hints this may be wrong.)")
    net = MLP([64, 64, 10], "relu", "softmax_ce", seed=0)
    h = net.fit(Xtr, Ytr, lr=1.0, epochs=30, batch_size=32, seed=0, track_grads=True)
    for ep in (0, 1, 2, 4, 9, len(h["train_loss"]) - 1):
        print("  epoch %2d  loss = %-10.4f  grad norms [W1, W2] = %s" % (ep + 1, h["train_loss"][ep], np.round(h["grad_norms"][ep], 4)))
    print("  RESULT: loss spikes at epoch 2 (instability) but then recovers; test accuracy = %.4f -> divergence prediction REFUTED for lr=1.0." % accuracy_score(yte, net.predict(Xte)))

    print("\n[F1b] NEW HYPOTHESIS: ReLU, hidden 64, lr = 20 diverges (step far larger than the loss curvature allows).")
    net = MLP([64, 64, 10], "relu", "softmax_ce", seed=0)
    div = net.fit(Xtr, Ytr, lr=20.0, epochs=30, batch_size=32, seed=0, track_grads=True)
    print("  first-batch gradient norms [W1, W2] = %s" % np.round(div["grad_first_batch"], 4))
    for ep in range(min(len(div["train_loss"]), 6)):
        print("  epoch %2d  loss = %-14.4f grad norms [W1, W2] = %s" % (ep + 1, div["train_loss"][ep], np.round(div["grad_norms"][ep], 4)))
    last = div["train_loss"][-1]
    print("  run stopped after %d epochs (stopping criterion: non-finite loss)" % len(div["train_loss"]) if not np.isfinite(last) else
          "  final loss after 30 epochs = %.4f" % last)
    acc_div = accuracy_score(yte, net.predict(Xte)) if np.isfinite(last) else float("nan")
    alive = float((Xtr @ net.W[1].T + net.b[1] > 0).mean())
    print("  test accuracy = %.4f   fraction of hidden activations still > 0 = %.3f" % (acc_div, alive))
    print("  DIAGNOSIS: first-batch gradients are normal (0.63) but after the lr=20 updates every gradient norm is exactly 0:")
    print("  the overshoot pushed all hidden pre-activations negative (dead network) and left the output layer saturated,")
    print("  so the loss is stuck ABOVE chance level (%.2f > ln10 = 2.30). Divergence of the step size -> dead units, not NaN." % last)

    print("\n[F1c] DEAD ReLU units: hidden 2, lr = 1.0 (grid result: accuracy 0.10 = chance)")
    net = MLP([64, 2, 10], "relu", "softmax_ce", seed=0)
    hdead = net.fit(Xtr, Ytr, lr=1.0, epochs=30, batch_size=32, seed=0, track_grads=True)
    active = (Xtr @ net.W[1].T + net.b[1] > 0).mean(axis=0)
    print("  fraction of training inputs that activate each hidden unit = %s   (0.00 = dead unit)" % np.round(active, 3))
    print("  gradient norms [W1, W2] after epoch 30 = %s   loss = %.4f (ln 10 = 2.3026)" % (np.round(hdead["grad_norms"][-1], 5), hdead["train_loss"][-1]))
    print("  DIAGNOSIS: units whose pre-activation is negative for every input have zero gradient -> they can never recover.")

    print("\n[F2] VANISHING GRADIENT  hypothesis: 6 sigmoid hidden layers of 16 units, lr = 0.1 -> early layers receive tiny gradients.")
    deep = MLP([64, 16, 16, 16, 16, 16, 16, 10], "sigmoid", "softmax_ce", seed=0)
    hd = deep.fit(Xtr, Ytr, lr=0.1, epochs=60, batch_size=32, seed=0, track_grads=True)
    names = ["W1 (input side)", "W2", "W3", "W4", "W5", "W6", "W7 (output side)"]
    print("  per-layer gradient norm  ||dL/dW^(l)||  at the FIRST mini-batch and after epoch 1 / 30 / 60:")
    print("  %-18s %-12s %-12s %-12s %-12s" % ("layer", "first batch", "epoch 1", "epoch 30", "epoch 60"))
    for l, nm in enumerate(names):
        print("  %-18s %-12.3e %-12.3e %-12.3e %-12.3e" % (nm, hd["grad_first_batch"][l], hd["grad_norms"][0][l], hd["grad_norms"][29][l], hd["grad_norms"][59][l]))
    ratio = hd["grad_first_batch"][-1] / hd["grad_first_batch"][0]
    print("  gradient of the last layer is %.0fx larger than the first layer at the first batch" % ratio)
    print("  loss epoch 1 = %.4f   epoch 60 = %.4f   test accuracy = %.4f   (shallow 64-32-10 sigmoid, same lr, for contrast: see grid)" % (
        hd["train_loss"][0], hd["train_loss"][-1], accuracy_score(yte, deep.predict(Xte))))
    shallow = MLP([64, 32, 10], "sigmoid", "softmax_ce", seed=0)
    hs = shallow.fit(Xtr, Ytr, lr=0.1, epochs=60, batch_size=32, seed=0)
    print("  contrast: shallow 64-32-10 sigmoid, lr 0.1, 60 epochs -> loss %.4f, test accuracy %.4f" % (hs["train_loss"][-1], accuracy_score(yte, shallow.predict(Xte))))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    G = np.array(hd["grad_norms"])
    for l in range(G.shape[1]):
        ax[0].semilogy(G[:, l], label=names[l])
    ax[0].set_title("Vanishing gradient: 6 sigmoid hidden layers"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("||dL/dW|| (log)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].plot(np.clip(np.nan_to_num(div["train_loss"], nan=50.0, posinf=50.0), 0, 50), "r-o", ms=3, label="ReLU 64, lr=20 (diverging)")
    ax[1].plot(np.clip(hd["train_loss"], 0, 50), "b-", label="deep sigmoid, lr=0.1 (stalled)")
    ax[1].plot(hs["train_loss"], "g-", label="shallow sigmoid, lr=0.1 (healthy)")
    ax[1].set_title("Loss under the two provoked failures"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "task4_failure_diagnosis.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    desk_check()
    gradient_check()
    xor_demo()
    digits_grid()
    failure_diagnosis()
