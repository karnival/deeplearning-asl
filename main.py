import numpy as np

import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Add, BatchNormalization, TimeDistributed, Average
from keras.layers import Conv3D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, optimizers 

from generator import DataGenerator

# Create network architecture.
conv_reg_w = 0.01
n_channels = 4
filter_pix = 3
filter_size = (filter_pix, filter_pix, filter_pix)

model = Sequential()
model.add(Conv3D(64, filter_size,
                 padding='same',
                 activation='relu', input_shape=(None, None, None, n_channels)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv3D(64, filter_size,
                 padding='same',
                 activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv3D(64, filter_size,
                 padding='same',
                 activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv3D(64, filter_size,
                 padding='same',
                 activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#model.add(Dense(1))
model.add(Conv3D(1, filter_size,
                 padding='same'))

#model.add(Dense(1, kernel_regularizer=regularizers.l2(0.1),
#                input_shape=(None, None, None, 1)))

# Training details.
batch_size = 3
epochs = 1000

opt = optimizers.Adam(lr=0.01)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

# Load high/low quality images.
d = 'data/'

partition = dict()
partition['train'] = ['92267', '48775', '74343', '35162', '27155', '67504', 
                      '56643', '98394', '71479', '71981', '17065', '46558', 
                      '39341', '59011', '34160', '95012', '45148', '66036', 
                      '80743', '60112', '14416', '53459', '68990', '22598', 
                      '38475', '13620', '85904', '36395', '65751', '42959', 
                      '76780', '52653', '46831', '80621', '74131', '65088', 
                      '16926', '22313', '78130', '47931', '33936', '91922', 
                      '18966', '74067', '42259', '81324', '76797', '15516', 
                      '82845', '96989', '81555', '92050', '52422', '80643', 
                      '23140', '83747', '45195', '81926', '49706', '35837', 
                      '62078', '37648', '98774', '94749', '37808', '49479', 
                      '39887', '49589', '72993', '13267', '21302', '23128', 
                      '35893', '62943', '78978', '44631', '28999', '66734', 
                      '32437', '85253', '60962', '48536', '25985', '21511', 
                      '88031', '92452', '36559', '69241', '36937', '69131', 
                      '65750', '46431', '58846', '50493', '71935', '88335', 
                      '85411', '66610', '30142', '85981', '79717', '14121', 
                      '48996', '24217', '30002', '79815', '47476', '25316', 
                      '43203', '44254', '96179', '48142', '43105', '83677', 
                      '96349', '47719', '63959', '50846', '78004', '31084', 
                      '11610', '45442', '38479', '67358']

partition['validation'] = ['82343', '26132', '98865', '36152', '40818', '66302', 
                           '24300', '36577', '55894', '51795', '70915', '47318', 
                           '82329', '45858', '43607']


params = {'dimns' : (96, 96, 47),
          'channels' : ('aslmean', 'aslstd', 'm0', 't1'),
          'data_dir' : d,
          'batch_size': batch_size}

training_generator = DataGenerator(**params).generate(partition['train'])
validation_generator = DataGenerator(**params).generate(partition['validation'])

filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir="logs/")
callbacks_list = [checkpoint, tensorboard]

# Fit!
model.fit_generator(generator=training_generator, epochs=epochs,
          steps_per_epoch=len(partition['train'])//batch_size,
          validation_data=validation_generator,
          validation_steps=len(partition['validation'])//batch_size,
          callbacks=callbacks_list)

model.save('overfitted_model.hd5')
