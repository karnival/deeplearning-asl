# ASL-DL

Enhancement of arterial spin labelling MR images using deep learning.

Current plan: CNN trained to map from (low-repetition, noisy and artifact-prone ASL images + M0 image + T1 image) to (high-quality, filtered/smoothed perfusion map).

Architecture ideas: 
* CNN
* GAN?
* Treat time (fourth dimension) as a fourth spatial dimension.

Loss function ideas: 
* Voxelwise loss
* Adversarial loss?
* Perceptual loss?

