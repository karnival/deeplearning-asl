import numpy as np

import keras
from keras.losses import mean_squared_error

def masked_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred * np.where(y_true != 0))
