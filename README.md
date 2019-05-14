# Adaptively Tuning a Convolutional Neural Network by Gating Process for Image Denoising (IEEE ICIP 2019)
## Abstract
This paper presents a new framework that controls feature maps of a convolutional neural network (CNN) according to the noise level such that the network can have different properties to different levels. Unlike the conventional non-blind approach which reloads all the parameters of CNN or switches to other CNNs for different noise levels, we adjust the CNN activation feature maps without changing the parameters at the test phase. For this, we additionally construct a noise level indicator network, which gives appropriate weighting values to the feature maps for the given situation. The noise level indicator network is so simple that it can be implemented as a low-dimensional look-up table at the test phase and thus does not increase the overall complexity. From the experiments on noise reduction, we can observe that the proposed method achieves better performance compared to the baseline network.
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
