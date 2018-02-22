import numpy as np

import keras
from keras.losses import mean_squared_error
from keras.backend import dot, not_equal, switch, zeros_like, get_value, set_value, cast, floatx

def masked_loss(y_true, y_pred):
    mask = cast(not_equal(y_true, 0), floatx())
    
    return mean_squared_error(y_true, dot(y_pred, mask))
