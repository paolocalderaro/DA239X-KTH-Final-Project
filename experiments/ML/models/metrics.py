import numpy as np
from sklearn.metrics import mean_squared_error

def aRRMSE(y_true, y_pred):
    # compute averaged real value y_bar
    y_bar = np.repeat(y_true.mean(axis=0).reshape(-1, 1).T, np.shape(y_true)[0], axis=0)

    MSE = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    std = mean_squared_error(y_true, y_bar, multioutput='raw_values')

    result = (np.sqrt(MSE / std)).mean()
    return result


def aRMSE(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred, multioutput='raw_values')

    n_sample = np.shape(y_true)[0]

    result = np.sqrt(MSE / n_sample).mean()
    return result


def multiRMSE(y_true, y_pred, y_label):
    RMSE = mean_squared_error(y_true, y_pred, multioutput='raw_values', squared=False)
    result = dict(zip(y_label,RMSE))
    return result


def multiMSE(y_true, y_pred, y_label):
    MSE = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    result = dict(zip(y_label,MSE))
    return result


def adj_r2(r2, X_test):
    n = np.shape(X_test)[0]
    p = np.shape(X_test)[1]
    return 1 - ((1-r2)*(n-1) / (n-p-1))


# use this for debug
if __name__ == "__main__":
   r2 = 0.45
   n = 180
   p = 12
   result = 1 -((1-r2)*(n-1) / (n-p-1))