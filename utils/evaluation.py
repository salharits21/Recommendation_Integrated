import numpy as np


def precision_recall_f1(actual, predicted):
    actual    = set(actual)
    predicted = set(predicted)

    hits = len(actual & predicted)

    precision = hits / len(predicted) if predicted else 0
    recall    = hits / len(actual)    if actual    else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

    return precision, recall, f1
