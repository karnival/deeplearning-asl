import numpy as np
import nibabel as nib

import keras
from keras.models import Sequential, load_model


# Load high/low quality images.
d = 'data/'

input = nib.load(d+'input.nii.gz').get_data()
input = input[:,:,:,1::2] - input[:,:,:,0::2] # difference images
input = np.mean(input, axis=3) # mean

# Net expects an array of images, with channels dimension on end.
prepared_input = input
prepared_input = np.expand_dims(prepared_input, axis=0)
prepared_input = np.expand_dims(prepared_input, axis=-1)

model = load_model('overfitted_model.hd5')

o1 = model.predict(prepared_input)

print('diff: ', np.nanmean(np.abs(o1[0,:,:,:,0] - input)))

o2 = model.predict(prepared_input + np.random.randn(*np.shape(prepared_input)))
print('diff: ', np.nanmean(np.abs(o2[0,:,:,:,0] - input)))

print('avg: ', np.nanmean(np.abs(input)))

in_img = nib.load(d+'input.nii.gz')
out_img = nib.Nifti1Image(np.squeeze(o1[0,:,:,:,0]), None, header=in_img.header)
out_img.to_filename(d+'test_out.nii.gz')
out_img2 = nib.Nifti1Image(np.squeeze(o2[0,:,:,:,0]), None, header=in_img.header)
out_img2.to_filename(d+'test_out2.nii.gz')
