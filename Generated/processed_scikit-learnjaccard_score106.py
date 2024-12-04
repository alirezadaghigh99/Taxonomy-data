import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

def jaccard_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division="warn"):
    def _jaccard_index(y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        if union == 0:
            if zero_division == "warn":
                print("Warning: Zero division encountered in Jaccard index calculation.")
                return 0.0
            else:
                return zero_division
        return intersection / union

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if average == 'binary':
        y_true = (y_true == pos_label).astype(int)
        y_pred = (y_pred == pos_label).astype(int)
        return _jaccard_index(y_true, y_pred)

    elif average in ['micro', 'macro', 'weighted', 'samples']:
        y_true_bin = label_binarize(y_true, classes=labels)
        y_pred_bin = label_binarize(y_pred, classes=labels)

        if average == 'micro':
            return _jaccard_index(y_true_bin.ravel(), y_pred_bin.ravel())

        elif average == 'macro':
            scores = [_jaccard_index(y_true_bin[:, i], y_pred_bin[:, i]) for i in range(len(labels))]
            return np.mean(scores)

        elif average == 'weighted':
            scores = [_jaccard_index(y_true_bin[:, i], y_pred_bin[:, i]) for i in range(len(labels))]
            weights = np.sum(y_true_bin, axis=0)
            return np.average(scores, weights=weights)

        elif average == 'samples':
            scores = [_jaccard_index(y_true_bin[i], y_pred_bin[i]) for i in range(y_true_bin.shape[0])]
            return np.mean(scores)

    else:
        raise ValueError("average has to be one of ['binary', 'micro', 'macro', 'weighted', 'samples']")

