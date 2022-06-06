import numpy as np


def revenue_gain(y_true, y_pred):
    """
    Return the score according to the Estimated revenues/ fitmotion gain matrix
    from the slides. Faster than the revenue_challenge.py function.
    labels: 1="happy", 2="angry", 3="sad", 4="relaxed"
    for sklearn use this:
        from sklearn.metrics import make_scorer
        score = make_scorer(revenue_gain, greater_is_better=True)
        score(classifier, X, y) # same as revenue_gain(y, classifier.predict(X))

    :param y_true: true labels as numpy array of int
    :param y_pred: predicted labels as numpy array of int
    :return: accumulates score over all predictions
    """
    # convert to int and remove 1 to use as index for gain_matrix
    y_true = y_true.astype(int) - 1
    y_pred = y_pred.astype(int) - 1

    gain_matrix = np.array([[5, -5, -5, 2],
                            [-5, 10, 2, -5],
                            [-5, 2, 10, -5],
                            [2, -5, -2, 5]])

    revenue = gain_matrix[y_true, y_pred].sum()

    return revenue
