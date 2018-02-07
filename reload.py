import numpy as np
import nibabel as nib

import keras
from keras.models import Sequential, load_model


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


# Net expects an array of images, with channels dimension on end.
prepared_input = input
prepared_input = np.expand_dims(prepared_input, axis=0)

model = load_model('overfitted_model.hd5')

o1 = model.predict(prepared_input)

#print('diff: ', np.nanmean(np.abs(o1[0,:,:,:,0] - input)))

o2 = model.predict(prepared_input + np.random.randn(*np.shape(prepared_input)))
#print('diff: ', np.nanmean(np.abs(o2[0,:,:,:,0] - input)))

print('avg: ', np.nanmean(np.abs(input)))

in_img = nib.load(d+'asl.nii.gz')
out_img = nib.Nifti1Image(np.squeeze(o1[0,:,:,:,0]), None, header=in_img.header)
out_img.to_filename(d+'test_out.nii.gz')
out_img2 = nib.Nifti1Image(np.squeeze(o2[0,:,:,:,0]), None, header=in_img.header)
out_img2.to_filename(d+'test_out2.nii.gz')
