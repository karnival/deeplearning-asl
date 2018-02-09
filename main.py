import numpy as np
import nibabel as nib

import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Add, BatchNormalization, TimeDistributed, Average
from keras.layers import Conv3D
from keras import regularizers, optimizers 

# Load high/low quality images.
d = 'data/'

asl = nib.load(d+'asl.nii.gz').get_data()
asl = asl[:,:,:,1::2] - asl[:,:,:,0::2] # difference images
asl = np.mean(asl, axis=3) # mean

m0 = nib.load(d+'calib.nii.gz').get_data()
t1 = nib.load(d+'struct.nii.gz').get_data()

# Different channel for each type of image.
input = np.zeros(np.shape(m0) + (3,))
input[:,:,:,0] = asl
input[:,:,:,1] = m0
input[:,:,:,2] = t1

x_train = np.expand_dims(input, axis=0)

output = nib.load(d+'output.nii.gz').get_data()

y_train = np.expand_dims(output, axis=0)
y_train = np.expand_dims(y_train, axis=-1)

x_test = x_train
y_test = y_train


# Create network architecture.
conv_reg_w = 0.01
n_channels = 3
filter_pix = 7
filter_size = (filter_pix, filter_pix, filter_pix)

model = Sequential()
model.add(Conv3D(64, filter_size,
                 padding='same',
                 activation='relu', input_shape=(None, None, None, n_channels)))
model.add(BatchNormalization())
model.add(Conv3D(64, filter_size,
                 padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv3D(64, filter_size,
                 padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv3D(64, filter_size,
                 padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv3D(1, filter_size,
                 padding='same'))

#model.add(Dense(1, kernel_regularizer=regularizers.l2(0.1),
#                input_shape=(None, None, None, 1)))

batch_size = 1
epochs = 1000

opt = optimizers.Adam(lr=0.01)

model.compile(loss='mean_squared_error', optimizer=opt)

# Fit!
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

model.save('overfitted_model.hd5')

print('size is: ', np.size(input))
print('avg is: ', np.nanmean(input.ravel()))
