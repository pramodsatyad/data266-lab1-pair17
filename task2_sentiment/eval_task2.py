import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

NEGATION = re.compile(r"\b(?:not|no|never)\b|n['’]t")
CONTRAST = re.compile(r"\b(?:but|however|although)\b")


def classification_metrics(y_true, y_pred):
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ["macro", "micro", "weighted"]:
        out[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return out


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def plot_confusion_matrix(cm, path, title="Confusion Matrix", class_names=("negative", "positive")):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def roc_auc(y_true, prob):
    return roc_auc_score(y_true, prob)


def pr_auc(y_true, prob):
    return average_precision_score(y_true, prob)


def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)


def brier_score(y_true, prob):
    return float(np.mean((np.asarray(prob) - np.asarray(y_true)) ** 2))


def reliability_bins(y_true, prob, n_bins=15):
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(prob, edges[1:-1])
    counts = np.bincount(idx, minlength=n_bins)
    sum_prob = np.bincount(idx, weights=prob, minlength=n_bins)
    sum_pos = np.bincount(idx, weights=y_true, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_prob = sum_prob / counts
        frac_pos = sum_pos / counts
    return mean_prob, frac_pos, counts


def expected_calibration_error(y_true, prob, n_bins=15):
    mean_prob, frac_pos, counts = reliability_bins(y_true, prob, n_bins)
    used = counts > 0
    return float(np.sum(counts[used] / counts.sum() * np.abs(frac_pos[used] - mean_prob[used])))


def plot_reliability(y_true, prob, path, title="Reliability Plot", n_bins=15):
    mean_prob, frac_pos, counts = reliability_bins(y_true, prob, n_bins)
    ece = expected_calibration_error(y_true, prob, n_bins)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    used = counts > 0
    ax.plot(mean_prob[used], frac_pos[used], marker="o", label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("share of positive reviews")
    ax.set_title(f"{title} (ECE {ece:.3f})")
    ax.legend()
    ax.grid(True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def metrics_from_counts(c):
    tn, fp, fn, tp = c
    n = tn + fp + fn + tp
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn > 0 else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_value = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return (tp + tn) / n, (f1_pos + f1_neg) / 2, mcc_value


def bootstrap_ci(y_true, y_pred, n_resamples=1000, seed=42):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    code = 2 * y_true + y_pred
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty((n_resamples, 3))
    for i in range(n_resamples):
        counts = np.bincount(code[rng.integers(0, n, n)], minlength=4)
        stats[i] = metrics_from_counts(counts)
    low, high = np.percentile(stats, [2.5, 97.5], axis=0)
    return {
        "accuracy": (low[0], high[0]),
        "f1_macro": (low[1], high[1]),
        "mcc": (low[2], high[2]),
    }


def mcnemar_exact(y_true, pred_a, pred_b):
    a_right = np.asarray(pred_a) == np.asarray(y_true)
    b_right = np.asarray(pred_b) == np.asarray(y_true)
    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    p_value = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"b": b, "c": c, "p_value": p_value}


def make_slices(texts, word_counts):
    words = np.asarray(word_counts)
    lower = [t.lower() for t in texts]
    return {
        "short": words < 50,
        "medium": (words >= 50) & (words <= 200),
        "long": words > 200,
        "has_negation": np.array([bool(NEGATION.search(t)) for t in lower]),
        "has_contrast": np.array([bool(CONTRAST.search(t)) for t in lower]),
    }


def slice_metrics(y_true, y_pred, slices):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for name, mask in slices.items():
        mask = np.asarray(mask, dtype=bool)
        out[f"{name}_n"] = int(mask.sum())
        if mask.sum() == 0:
            out[f"{name}_f1_macro"] = np.nan
            out[f"{name}_error_rate"] = np.nan
        else:
            out[f"{name}_f1_macro"] = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            out[f"{name}_error_rate"] = float((y_true[mask] != y_pred[mask]).mean())
    return out


def evaluate_all(y_true, prob, pred=None, slices=None, n_resamples=1000, seed=42):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int) if pred is None else np.asarray(pred, dtype=int)

    out = classification_metrics(y_true, pred)
    tn, fp, fn, tp = get_confusion_matrix(y_true, pred).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    out["roc_auc"] = roc_auc(y_true, prob)
    out["pr_auc"] = pr_auc(y_true, prob)
    out["mcc"] = mcc(y_true, pred)
    out["brier"] = brier_score(y_true, prob)
    out["ece"] = expected_calibration_error(y_true, prob)
    for name, (low, high) in bootstrap_ci(y_true, pred, n_resamples, seed).items():
        out[f"{name}_ci_low"] = low
        out[f"{name}_ci_high"] = high
    if slices is not None:
        out.update(slice_metrics(y_true, pred, slices))
    return out
