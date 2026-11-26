import numpy as np
from sklearn.metrics import f1_score


def top1_and_macro_f1(labels, predictions):
    # top-1 accuracy in % and macro-F1 (0-1)
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    return 100.0 * float((labels == predictions).mean()), float(f1_score(labels, predictions, average="macro", zero_division=0))


def macro_f1_from_confusion(confusion):
    # macro-F1 of a (K, K) confusion matrix (rows = true class); a class without recordings counts as 0, as in sklearn with zero_division=0
    true_positive = np.diag(confusion).astype(float)
    precision = np.divide(true_positive, confusion.sum(0), out=np.zeros_like(true_positive), where=confusion.sum(0) > 0)
    recall = np.divide(true_positive, confusion.sum(1), out=np.zeros_like(true_positive), where=confusion.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_positive), where=(precision + recall) > 0)
    return float(f1.mean())


def group_bootstrap_gain(labels, base_predictions, new_predictions, groups, num_classes, n_boot=5000, seed=2026):
    # Gain of new_predictions over base_predictions (top-1 in points, macro-F1) with 95% percentile intervals from resampling whole groups
    # (wells, compounds, ...), because recordings of one group are not independent.
    labels, base_predictions, new_predictions = map(np.asarray, (labels, base_predictions, new_predictions))
    group_ids, group_index = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    n_groups = len(group_ids)
    confusion_base = np.zeros((n_groups, num_classes, num_classes))
    confusion_new = np.zeros((n_groups, num_classes, num_classes))
    for confusion, predictions in ((confusion_base, base_predictions), (confusion_new, new_predictions)):
        np.add.at(confusion, (group_index, labels, predictions), 1)
    rng = np.random.default_rng(seed)
    gain_top1, gain_f1 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        weights = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups).astype(float)
        resampled_base, resampled_new = np.tensordot(weights, confusion_base, 1), np.tensordot(weights, confusion_new, 1)
        gain_top1[b] = 100.0 * (np.trace(resampled_new) - np.trace(resampled_base)) / max(resampled_base.sum(), 1)
        gain_f1[b] = macro_f1_from_confusion(resampled_new) - macro_f1_from_confusion(resampled_base)
    (top1_base, f1_base), (top1_new, f1_new) = top1_and_macro_f1(labels, base_predictions), top1_and_macro_f1(labels, new_predictions)

    def interval(values):
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    return dict(n_groups=n_groups, top1_gain=top1_new - top1_base, top1_ci=interval(gain_top1), f1_gain=f1_new - f1_base, f1_ci=interval(gain_f1),
                p_top1_le_0=float((gain_top1 <= 0).mean()))


def exact_mcnemar(labels, base_predictions, new_predictions):
    # exact two-sided McNemar test on the recordings where exactly one of the two predictions is correct; returns (fixed, broken, p)
    from scipy.stats import binomtest
    labels = np.asarray(labels)
    fixed = int(((base_predictions != labels) & (new_predictions == labels)).sum())
    broken = int(((base_predictions == labels) & (new_predictions != labels)).sum())
    p_value = 1.0 if fixed + broken == 0 else float(binomtest(fixed, fixed + broken, 0.5).pvalue)
    return fixed, broken, p_value
