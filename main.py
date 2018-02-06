import numpy as np
import nibabel as nib

import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Add
from keras.layers import Conv3D
from keras import regularizers, optimizers 

# Load high/low quality images.
d = 'data/'

input = nib.load(d+'input.nii.gz').get_data()
input = input[:,:,:,1::2] - input[:,:,:,0::2] # difference images
input = np.mean(input, axis=3) # mean

output = nib.load(d+'output.nii.gz').get_data()

x_train = np.expand_dims(input, axis=0)
x_train = np.expand_dims(x_train, axis=-1)
y_train = np.expand_dims(output, axis=0)
y_train = np.expand_dims(y_train, axis=-1)

x_test = x_train
y_test = y_train


# Create network architecture.
model = Sequential()

model.add(Conv3D(12, (7, 7, 7), input_shape=(None, None, None, 1),
                 padding='same', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(1, kernel_regularizer=regularizers.l2(0.1),
                input_shape=(None, None, None, 1)))
print(model.output_shape)

batch_size = 1
epochs = 2000

opt = optimizers.Adam(lr=0.1)

model.compile(loss='mean_squared_error', optimizer=opt)

# Fit!
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

print('size is: ', np.size(input))
print('avg is: ', np.nanmean(input.ravel()))
