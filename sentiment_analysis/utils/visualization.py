from typing import Optional, List
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from numpy import newaxis, ndarray
from pandas import Series

def plot_confusion_matrix(y_true: Series, y_pred: ndarray, model_name: str, classes: Optional[List[int]] = None, normalize: Optional[bool] = False) -> None:
    """
    Plots the confusion matrix for a given model's predictions.

    Parameters:
    - y_true (pandas.Series): The true labels.
    - y_pred (numpy.ndarray): The predicted labels.
    - model_name (str): The name of the model.
    - classes (Optional[List[int]]): The list of class labels. Default is [0, 1].
    - normalize (Optional[bool]): Whether to normalize the confusion matrix. Default is False.

    Returns:
    - None
    """
    if classes is None:
        classes = [0, 1]

    cfm = confusion_matrix(y_true, y_pred)

    if normalize:
        cfm = cfm.astype('float') / cfm.sum(axis=1)[:, newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    heatmap(cfm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)

    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()