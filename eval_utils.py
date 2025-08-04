from scipy.optimize import linear_sum_assignment
import numpy as np


def clustering_accuracy(true_labels, predicted_labels):
    D = max(true_labels.max(), predicted_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(true_labels.shape[0]):
        w[true_labels[i], predicted_labels[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / true_labels.shape[0]


