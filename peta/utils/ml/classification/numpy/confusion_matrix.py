import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import Tensor


def _to_np_array(x: Tensor):
    if isinstance(x, np.ndarray):
        pass
    elif isinstance(x, Tensor):
        x = x.detach().cpu()
    else:
        x = np.array(x)
    return x


def plot_confusion_matrix(predictions: np.ndarray, targets: np.ndarray):
    predictions = _to_np_array(predictions)
    targets = _to_np_array(targets)
    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    return disp.figure_, disp.ax_
