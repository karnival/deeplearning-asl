import numpy as np
import nibabel as nib

import keras

class DataGenerator(object):
    def __init__(self, dimns, channels, batch_size, data_dir):
        self.dim_0 = dimns[0]
        self.dim_1 = dimns[1]
        self.dim_2 = dimns[2]

        self.channels = channels
        self.batch_size = batch_size

        self.data_dir = data_dir

    def generate(self, list_IDs):
        while 1:
            indices = self._get_exploration_order(list_IDs)

            imax = int(len(indices) / self.batch_size)

            for i in range(imax):
                to_use_IDs = [list_IDs[k] for k in
                              indices[i*self.batch_size:(i+1)*self.batch_size]]

                x, y = self._data_generation(to_use_IDs)

                yield x, y

    def _get_exploration_order(self, list_IDs):
        indices = np.arange(len(list_IDs))

        np.random.shuffle(indices)

        return indices

    def _data_generation(self, to_use_IDs):
        x = np.empty((self.batch_size, self.dim_0, self.dim_1, self.dim_2,
                      len(self.channels)))
        y = np.empty((self.batch_size, self.dim_0, self.dim_1, self.dim_2, 1))

        for i, id in enumerate(to_use_IDs):
            for j, chan in enumerate(self.channels):
                tmp = nib.load(self.data_dir + '/' + id + '/in_' + chan + '.nii.gz')
                tmp = tmp.get_data()
                x[i,:,:,:,j] = tmp

            y[i,:,:,:,0] = nib.load(self.data_dir + '/' + id + '/out.nii.gz').get_data()

        return x, y
