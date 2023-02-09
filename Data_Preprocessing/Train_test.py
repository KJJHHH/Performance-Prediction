import numpy as np

def loss_f(y_pred, data_ytest):
    return np.abs(y_pred - np.array(data_ytest).reshape(-1, 1)).mean()
