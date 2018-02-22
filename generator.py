import numpy as np
import scipy.ndimage as sp
import nibabel as nib

import keras

class DataGenerator(object):
    def __init__(self, dimns, channels, batch_size, data_dir, n_to_use=10):
        self.dim_0 = dimns[0]
        self.dim_1 = dimns[1]
        self.dim_2 = dimns[2]

        self.channels = channels
        self.batch_size = batch_size

        self.data_dir = data_dir
        self.n_to_use = n_to_use;

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
        augmentation = False

        x = np.empty((self.batch_size, self.dim_0, self.dim_1, self.dim_2,
                      len(self.channels)))
        y = np.empty((self.batch_size, self.dim_0, self.dim_1, self.dim_2, 1))

        for i, id in enumerate(to_use_IDs):
            # take a subset of ASL DIs, for finding aslmean and aslstd
            asl = nib.load(self.data_dir + '/' + id + '/asl_res_moco.nii.gz')
            asl = asl.get_data()
            asl = asl[:,:,:,0::2] - asl[:,:,:,1::2]

            bmask = nib.load(self.data_dir + '/' + id +'/bmask_t1.nii.gz').get_data()

            asl[np.where(bmask==0)] = 0

            if augmentation:
                n_vols = np.shape(asl)[-1]
                vols_to_use = np.random.choice(n_vols, self.n_to_use, replace=False)

                asls_to_use = asl[:,:,:,vols_to_use]
            else:
                asls_to_use = asl[:,:,:,0:9]

            for j, chan in enumerate(self.channels):
                if chan is 'aslmean':
                    tmp = np.nanmean(asls_to_use, 3)
                elif chan is 'aslstd':
                    tmp = np.nanstd(asls_to_use, 3)
                else:
                    tmp = nib.load(self.data_dir + '/' + id + '/in_' + chan + '.nii.gz')
                    tmp = tmp.get_data()
                    tmp[np.where(bmask==0)] = 0

                if chan is 'm0':
                    m0_unscaled = tmp

                tmp = (tmp - 10) / 10 # approx z-scaling for PWIs
                tmp[np.where(bmask==0)] = 0
                x[i,:,:,:,j] = tmp

            g_truth = nib.load(self.data_dir + '/' + id +
                                    '/asl_res_moco_filtered_mean.nii.gz').get_data()
            g_truth[np.where(bmask==0)] = 0

            # normalise ground truth by relevant inputs
            #same_scale_as_gt = asl_unscaled #/ m0_unscaled
            #same_scale_as_gt[np.isfinite(same_scale_as_gt)==False] = 0

            norm_mean = 10
            norm_std = 10

            g_truth = (g_truth - norm_mean) / norm_std
            g_truth[np.where(bmask==0)] = 0
            y[i,:,:,:,0] = g_truth 

        if augmentation:
            # translation
            t = np.random.uniform(-5, 5, size=2)
            t = np.append(t, np.random.uniform(-2.5, 2.5)) # through-plane

            for i in range(x.shape[0]):
                y[i,:,:,:,0] = sp.interpolation.shift(y[i,:,:,:,0], t)

                for j in range(x.shape[-1]):
                    x[i,:,:,:,j] = sp.interpolation.shift(x[i,:,:,:,j], t)

            # rotation
            r = np.random.uniform(-20, 20, size=3)


            # this rotation is around the origin, so need to translate after
            rot = lambda inp, ang, ax: sp.interpolation.rotate(inp, ang, ax, reshape=False)
            all_rots = lambda inp: rot(rot(rot(inp,
                                       r[0], (0, 1)),
                                           r[1], (0, 2)),
                                               r[2], (1, 2))

            for i in range(x.shape[0]):
                tmpy = y[i,:,:,:,0]
                center = sp.measurements.center_of_mass(tmpy)

                tmpy = all_rots(tmpy)

                for j in range(x.shape[-1]):
                    tmpx = x[i,:,:,:,j]
                    tmpx = all_rots(tmpx)
                    x[i,:,:,:,j] = tmpx

                rot_center = sp.measurements.center_of_mass(tmpy)

                # translate back to make rotation around center of mass
                transl = np.subtract(center, rot_center)
                tmpy = sp.interpolation.shift(tmpy, transl)
                y[i,:,:,:,0] = tmpy

                for j in range(x.shape[-1]):
                    x[i,:,:,:,j] = sp.interpolation.shift(x[i,:,:,:,j],
                                                                  transl)

        return x, y
