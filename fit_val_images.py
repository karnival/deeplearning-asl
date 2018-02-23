import numpy as np
import nibabel as nib
import os.path

import keras
from keras.models import load_model
import masked_loss

m_loss = masked_loss.masked_loss_factory()

model = load_model('aslmean-only.hdf5',
                   custom_objects={'masked mse': m_loss})

def normalise(im, mask, mu, std):
    im = (im - mu) / std

    im[np.where(mask == 0)] = 0

    return im

d = 'data/'

subjs = ['82343', '26132', '98865', '36152', '40818', '66302', 
         '24300', '36577', '55894', '51795', '70915', '47318', 
         '82329', '45858', '43607']

for s in subjs:
    bmask = nib.load(os.path.join(d,s,'bmask_t1.nii.gz')).get_data()
    loader = lambda fn: nib.load(os.path.join(d,s,fn)).get_data()

    asls = loader('asl_res_moco.nii.gz')
    asls_filt = loader('asl_res_moco.nii.gz_filtered.nii.gz')

    diffs = asls[:,:,:,0::2] - asls[:,:,:,1::2]

    mean_1 = np.nanmean(diffs[:,:,:,0:10], axis=-1)
    mean_1 = normalise(mean_1, bmask, 10, 10)
    mean_2 = np.nanmean(diffs[:,:,:,10:20], axis=-1)
    mean_2 = normalise(mean_2, bmask, 10, 10)

    std_1 = np.nanstd(diffs[:,:,:,0:10], axis=-1)
    std_1 = normalise(std_1, bmask, 7, 6) 
    std_2 = np.nanstd(diffs[:,:,:,10:20], axis=-1)
    std_2 = normalise(std_2, bmask, 7, 6) 

    in_data = np.empty((2,) + np.shape(mean_1) + (1,))

    in_data[0,:,:,:,0] = mean_1
    #in_data_1[0,:,:,:,1] = std_1

    in_data[1,:,:,:,0] = mean_2
    #in_data_1[1,:,:,:,1] = std_2

    norm_std = 10
    norm_mean = 10

    out = (model.predict(in_data) * norm_std) + norm_mean
    out[:,bmask==0,0] = 0

    outfile = nib.load(os.path.join(d,s,'bmask_t1.nii.gz'))
    out_img1 = nib.Nifti1Image(np.squeeze(out[0,:,:,:,0]), None, header=outfile.header)
    out_img1.to_filename(os.path.join(d, s, 'dl_fitted_1.nii.gz'))
    out_img2 = nib.Nifti1Image(np.squeeze(out[1,:,:,:,0]), None, header=outfile.header)
    out_img2.to_filename(os.path.join(d, s, 'dl_fitted_2.nii.gz'))
