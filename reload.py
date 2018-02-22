import numpy as np
import nibabel as nib

import keras
from keras.models import Sequential, load_model
import masked_loss

m_loss = masked_loss.masked_loss_factory()

def normalise(im, mask):
    im = (im - 10) / 10

    im[np.where(mask == 0)] = 0

    return im

# Load high/low quality images.
d = 'data_lores/11610/'

bmask = nib.load(d+'bmask_t1.nii.gz').get_data()
#bmask = bmask != 0

loader = lambda fn: normalise(nib.load(d+fn).get_data(), bmask)

aslmean = loader('least_squares.nii.gz')
#aslstd = loader('aslstd.nii.gz')
#m0 = loader('in_m0.nii.gz')
#t1 = loader('in_t1.nii.gz')

# calculate same scale as GT
aslmean_tmp = nib.load(d+'least_squares.nii.gz').get_data()
#m0_tmp = nib.load(d+'in_m0.nii.gz').get_data()

same_scale = aslmean_tmp #/ m0_tmp
#same_scale[np.isfinite(same_scale) == False] = 0

norm_mean = 10
norm_std = 10

# Net expects an array of images, with channels dimension on end.
prepared_input = np.empty(np.shape(aslmean) + (1,))
prepared_input[:,:,:,0] = aslmean
#prepared_input[:,:,:,1] = aslstd
#prepared_input[:,:,:,2] = m0
#prepared_input[:,:,:,3] = t1
prepared_input = np.expand_dims(prepared_input, axis=0)

truth = nib.load(d+'asl_res_moco_filtered_mean.nii.gz').get_data()
truth[np.where(bmask==0)] = 0

#model = load_model('overfitted_model.hd5')
model = load_model('weights-improvement-2580-4.77E-03.hdf5',
                   custom_objects={'masked mse': m_loss})

o1 = (model.predict(prepared_input) * norm_std) + norm_mean
o1[:,bmask==0,0] = 0
#o1 = model.predict(prepared_input)

#print('diff: ', np.nanmean(np.abs(o1[0,:,:,:,0] - truth)))

#o2 = model.predict(prepared_input + np.random.randn(*np.shape(prepared_input)))
#print('diff: ', np.nanmean(np.abs(o2[0,:,:,:,0] - truth)))

sq_errs = (o1[0,:,:,:,0] - truth)**2

print('sq error opt: ', np.nanmean(sq_errs[bmask != 0]))

in_img = nib.load(d+'least_squares.nii.gz')
out_img = nib.Nifti1Image(np.squeeze(o1[0,:,:,:,0]), None, header=in_img.header)
out_img.to_filename(d+'test_out.nii.gz')
#out_img2 = nib.Nifti1Image(np.squeeze(o2[0,:,:,:,0]), None, header=in_img.header)
#out_img2.to_filename(d+'test_out2.nii.gz')

