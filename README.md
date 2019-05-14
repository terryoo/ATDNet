# Adaptively Tuning a Convolutional Neural Network by Gating Process for Image Denoising (IEEE ICIP 2019)
## Abstract
## Network Architecture

# Adaptively Tuning a Convolutional Neural Network by Gate Process for Image Denoising (IEEE Access 2019)
This JOURNAL paper is the extended version of ICIP 2019 Conference Paper above. 
The most significant change from the ICIP version is to make the network spatially adaptive. In realistic environments, the noise is usually spatially variant, i.e., the noise variance changes gradually or abruptly depending on the regions. Hence, for dealing with the realistic noises, we made the noise-level estimator spatially adaptive such that it estimates pixel-wise noise variance. Also, the network is trained more sophisticatedly such as data augmentation and bigger patch size than ICIP 2019. As a result, this version works better for real-noises and spatially varying noises.

## Abstract
## Network Architecture
## Experimental Results

### Gaussian Denoising Results
<img src = "/figs/figure_gaussian_gray.PNG" width="900">
The comparison of gray channel denoisers. 

<img src = "/figs/figure_gaussian_color.PNG" width="900">
The comparison of color channel denoisers. 

### Real Noise Denoising Results
<img src = "/figs/figure_real.PNG" width="900">
The comparison of gray channel denoisers. 
