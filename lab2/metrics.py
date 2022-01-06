import numpy as np
import sklearn.metrics as skm


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes = None) -> np.ndarray:
    """"
    Computes the confusion matrix from labels (y_true) and predictions (y_pred).
    The matrix columns represent the prediction labels and the rows represent the ground truth labels.
    The confusion matrix is always a 2-D array of shape `[num_classes, num_classes]`,
    where `num_classes` is the number of valid labels for a given classification task.
    The arguments y_true and y_pred must have the same shapes in order for this function to work

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    conf_mat = None
    # TODO your code here - compute the confusion matrix
    # even here try to use vectorization, so NO for loops

    # 0. if the number of classes is not provided, compute it based on the y_true and y_pred arrays
    if not num_classes:
        num_classes = len(set(y_true))
    # 1. create a confusion matrix of shape (num_classes, num_classes) and initialize it to 0
    conf_mat = np.zeros((num_classes, num_classes))
    # 2. use argmax to get the maximal prediction for each sample
    # hint: you might find np.add.at useful: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
    np.add.at(conf_mat, (y_true, y_pred), 1)
    # end TODO your code here
    return conf_mat


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None) -> float:
    """"
    Computes the precision score.
    For binary classification, the precision score is defined as the ratio tp / (tp + fp)
    where tp is the number of true positives and fp the number of false positives.

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    precision = 0
    # TODO your code here
    conf_mat = confusion_matrix(y_true, y_pred)
    sum_rows = np.sum(conf_mat, axis=0)  # up-down
    conf_mat_divided = np.divide(conf_mat, sum_rows)
    precision = np.diagonal(conf_mat_divided)
    
    # Better version:
    # precision = np.diagonal(conf_mat) / np.sum(conf_mat, axis=0)
    # return the average of the array above and change in assert
    # end TODO your code here
    return precision


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None)  -> float:
    """"
    Computes the recall score.
    For binary classification, the recall score is defined as the ratio tp / (tp + fn)
    where tp is the number of true positives and fn the number of false negatives

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    recall = None
    # TODO your code here
    conf_mat = confusion_matrix(y_true, y_pred)
    sum_cols = np.sum(conf_mat, axis=1)  # left-right
    conf_mat_divided = np.divide(conf_mat, sum_cols)
    recall = np.diagonal(conf_mat_divided)
    # end TODO your code here
    return recall


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc_score = 0
    # TODO your code here
    # remember, use vectorization, so no for loops
    # hint: you might find np.trace useful here https://numpy.org/doc/stable/reference/generated/numpy.trace.html
    conf_mat = confusion_matrix(y_true, y_pred)
    acc_score = np.trace(conf_mat) / np.sum(conf_mat)
    # end TODO your code here
    return acc_score


if __name__ == '__main__':
    # TODO your tests here
    # CONFUSION MATRIX TESTS
    y_true = [0,1,2,3,4,5,6,7,8,9]
    y_pred = [0,0,2,3,4,5,6,7,8,9]
    conf_mat = confusion_matrix(y_true, y_pred)
    sk_conf_mat = skm.confusion_matrix(y_true, y_pred)
    assert (np.allclose(conf_mat, sk_conf_mat))

    random_array_true = np.random.randint(10, size=100)
    random_array_pred = np.random.randint(10, size=100)
    conf_mat = confusion_matrix(random_array_true, random_array_pred)
    sk_conf_mat = skm.confusion_matrix(random_array_true, random_array_pred)
    assert (np.allclose(conf_mat, sk_conf_mat))

    # PRECISION TESTS
    y_true = [0,1,2]
    y_pred = [0,1,2]
    prc = precision_score(y_true, y_pred)
    sk_prc = skm.precision_score(y_true, y_pred, average=None)
    assert (np.allclose(prc, sk_prc))

    prc = precision_score(random_array_true, random_array_pred)
    sk_prc = skm.precision_score(random_array_true, random_array_pred, average=None)
    assert (np.allclose(prc, sk_prc))

    # RECALL TESTS
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    rcl = recall_score(y_true, y_pred)
    sk_rcl = skm.recall_score(y_true, y_pred, average=None)
    assert (np.allclose(rcl, sk_rcl))

    rcl = recall_score(random_array_true, random_array_pred)
    sk_rcl = skm.recall_score(random_array_true, random_array_pred, average=None)
    assert (np.allclose(rcl, sk_rcl))

    # ACCURACY TESTS
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    acc = accuracy_score(y_true, y_pred)
    sk_acc = skm.accuracy_score(y_true, y_pred)
    assert (np.allclose(acc, sk_acc))

    acc = accuracy_score(random_array_true, random_array_pred)
    sk_acc = skm.accuracy_score(random_array_true, random_array_pred)
    assert (np.allclose(acc, sk_acc))

    # add some test for your code.
    # you could use the sklean.metrics module (with macro averaging to check your results)
