import numpy as np
import nibabel as nib

import keras
from keras.models import Sequential, load_model


# Load high/low quality images.
d = 'data/11610/'

aslmean = nib.load(d+'least_squares.nii.gz').get_data()
aslstd = nib.load(d+'aslstd.nii.gz').get_data()
m0 = nib.load(d+'in_m0.nii.gz').get_data()
t1 = nib.load(d+'in_t1.nii.gz').get_data()

# Net expects an array of images, with channels dimension on end.
prepared_input = np.empty(np.shape(aslmean) + (4,))
prepared_input[:,:,:,0] = aslmean
prepared_input[:,:,:,1] = aslstd
prepared_input[:,:,:,2] = m0
prepared_input[:,:,:,3] = t1
prepared_input = np.expand_dims(prepared_input, axis=0)

truth = aslmean

#model = load_model('overfitted_model.hd5')
model = load_model('weights-improvement-39-0.00.hdf5')

o1 = model.predict(prepared_input)

print('diff: ', np.nanmean(np.abs(o1[0,:,:,:,0] - truth)))

o2 = model.predict(prepared_input + np.random.randn(*np.shape(prepared_input)))
print('diff: ', np.nanmean(np.abs(o2[0,:,:,:,0] - truth)))

print('avg: ', np.nanmean(np.abs(truth)))

in_img = nib.load(d+'least_squares.nii.gz')
out_img = nib.Nifti1Image(np.squeeze(o1[0,:,:,:,0]), None, header=in_img.header)
out_img.to_filename(d+'test_out.nii.gz')
out_img2 = nib.Nifti1Image(np.squeeze(o2[0,:,:,:,0]), None, header=in_img.header)
out_img2.to_filename(d+'test_out2.nii.gz')
