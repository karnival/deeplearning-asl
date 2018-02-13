import numpy as np

import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Add, BatchNormalization, TimeDistributed, Average
from keras.layers import Conv3D
from keras import regularizers, optimizers 

from generator import DataGenerator

# Create network architecture.
conv_reg_w = 0.01
n_channels = 4
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
#model.add(Dense(1))
model.add(Conv3D(1, filter_size,
                 padding='same'))

#model.add(Dense(1, kernel_regularizer=regularizers.l2(0.1),
#                input_shape=(None, None, None, 1)))

# Training details.
batch_size = 1
epochs = 300

opt = optimizers.Adam(lr=0.01)

model.compile(loss='mean_squared_error', optimizer=opt)

# Load high/low quality images.
d = 'data/'
batch_size = 1

partition = dict()
partition['train'] = ['1']
partition['validation'] = ['1']

params = {'dimns' : (24, 24, 5),
          'channels' : ('aslmean', 'aslstd', 'm0', 't1'),
          'data_dir' : d,
          'batch_size': batch_size}

training_generator = DataGenerator(**params).generate(partition['train'])
validation_generator = DataGenerator(**params).generate(partition['validation'])

# Fit!
model.fit_generator(generator=training_generator, epochs=epochs,
          steps_per_epoch=len(partition['train'])//batch_size,
          validation_data=validation_generator,
          validation_steps=len(partition['validation'])//batch_size)

model.save('overfitted_model.hd5')
