import numpy as np

import keras
from keras.losses import mean_squared_error
import keras.backend as K
#from keras.backend import dot, not_equal, switch, zeros_like, get_value, set_value, cast, floatx, cast_to_floatx

def masked_loss_factory():
    def f(y_true, y_pred):
        mask = K.not_equal(y_true, 0)
        mask = K.cast(mask, K.floatx())
        masked_squared_err = K.square(mask * (y_true - y_pred))
        masked_mse = K.sum(masked_squared_err) / K.sum(mask)

        return masked_mse

    f.__name__ = 'masked_mse'
    return f
