import numpy as np
import nibabel as nib

import keras
from keras.models import Sequential, load_model

def normalise(im, mask):
    #im = (im - np.nanmean(im[mask])) / np.nanstd(im[mask])

    im[mask == 0] = 0

    return im

# Load high/low quality images.
d = 'data/11610/'

bmask = nib.load(d+'bmask_t1.nii.gz').get_data()
#bmask = bmask != 0

loader = lambda fn: normalise(nib.load(d+fn).get_data(), bmask)

aslmean = loader('least_squares.nii.gz')
aslstd = loader('aslstd.nii.gz')
m0 = loader('in_m0.nii.gz')
t1 = loader('in_t1.nii.gz')

# calculate same scale as GT
aslmean_tmp = nib.load(d+'least_squares.nii.gz').get_data()
m0_tmp = nib.load(d+'in_m0.nii.gz').get_data()

#same_scale = aslmean_tmp / m0_tmp
#same_scale[np.isfinite(same_scale) == False] = 0

#norm_mean = np.nanmean(same_scale[bmask])
#norm_std = np.nanstd(same_scale[bmask])

# Net expects an array of images, with channels dimension on end.
prepared_input = np.empty(np.shape(aslmean) + (4,))
prepared_input[:,:,:,0] = aslmean
prepared_input[:,:,:,1] = aslstd
prepared_input[:,:,:,2] = m0
prepared_input[:,:,:,3] = t1
prepared_input = np.expand_dims(prepared_input, axis=0)

truth = aslmean

#model = load_model('overfitted_model.hd5')
model = load_model('run_where_it_learns_to_mask_still_maybe_carry_on/weights-improvement-31-1.13E-05.hdf5')

#o1 = (model.predict(prepared_input) + norm_mean) * norm_std
o1 = model.predict(prepared_input)

#print('diff: ', np.nanmean(np.abs(o1[0,:,:,:,0] - truth)))

#o2 = model.predict(prepared_input + np.random.randn(*np.shape(prepared_input)))
#print('diff: ', np.nanmean(np.abs(o2[0,:,:,:,0] - truth)))

print('avg: ', np.nanmean(np.abs(truth)))

in_img = nib.load(d+'least_squares.nii.gz')
out_img = nib.Nifti1Image(np.squeeze(o1[0,:,:,:,0]), None, header=in_img.header)
out_img.to_filename(d+'test_out.nii.gz')
#out_img2 = nib.Nifti1Image(np.squeeze(o2[0,:,:,:,0]), None, header=in_img.header)
#out_img2.to_filename(d+'test_out2.nii.gz')

