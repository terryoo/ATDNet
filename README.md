# Adaptively Tuning a Convolutional Neural Network by Gating Process for Image Denoising (IEEE ICIP 2019)
## Test Code
[**ICIP Code**](https://github.com/terryoo/ATDNet/tree/master/icip)

[**Trained Model**](https://drive.google.com/drive/folders/1KjxUG0JbiiLxzTw0CVh5hLf6MGMU8x0h?usp=sharing)


## Abstract
This paper presents a new framework that controls feature maps of a convolutional neural network (CNN) according to the noise level such that the network can have different properties to different levels. Unlike the conventional non-blind approach which reloads all the parameters of CNN or switches to other CNNs for different noise levels, we adjust the CNN activation feature maps without changing the parameters at the test phase. For this, we additionally construct a noise level indicator network, which gives appropriate weighting values to the feature maps for the given situation. The noise level indicator network is so simple that it can be implemented as a low-dimensional look-up table at the test phase and thus does not increase the overall complexity. From the experiments on noise reduction, we can observe that the proposed method achieves better performance compared to the baseline network.


## Network Architecture
<img src = "/figs/figure_icip.png" width="900">
The architecture of ATDNet
<img src = "/figs/figure_icip_resblock.png" width="450">
The proposed Gate-ResBlock

## Experimental Results
<img src = "/figs/icip_table.PNG" width="900">

# Adaptively Tuning a Convolutional Neural Network by Gate Process for Image Denoising (IEEE Access 2019)
This JOURNAL paper is the extended version of ICIP 2019 Conference Paper above. 
The most significant change from the ICIP version is to make the network spatially adaptive. In realistic environments, the noise is usually spatially variant, i.e., the noise variance changes gradually or abruptly depending on the regions. Hence, for dealing with the realistic noises, we made the noise-level estimator spatially adaptive such that it estimates pixel-wise noise variance. Also, the network is trained more sophisticatedly such as data augmentation and bigger patch size than ICIP 2019. As a result, this version works better for real-noises and spatially varying noises.

## Paper
[**IEEE Access Paper**](https://ieeexplore.ieee.org/document/8717639)

## Test Code
[**Access Code**]

[**Trained Model**]

## Abstract
The conventional image denoising methods based on the convolutional neural network (CNN) focus on the non-blind training, and hence many networks are required to cope with various noise levels at the test. Although there are blind training methods that deal with multiple noise levels with a single network, their performance gain is generally lower than the non-blind ones, especially at low noise levels. In this paper, we propose a new denoising scheme that controls the feature maps of a single denoising network according to the noise level at the test phase, without changing the network parameters. This is achieved by employing a gating scheme where the feature maps of the denoising network are multiplied with appropriate weights from a gate-weight generating network which is trained along with the denoising network. We train the overall network on a wide range of noise level such that the proposed method can be used for both blind and non-blind cases. Experiments show that the proposed system yields better denoising performance than the other CNN-based methods, especially for the untrained noise levels. Finally, it is shown that the proposed system can manage spatially variant unknown noises and real noises without changing the whole CNN parameters.

## Network Architecture
<img src = "/figs/figure_access.png" width="900">
The architecture of ATDNet
<img src = "/figs/figure_access_block.png" width="450">
The proposed Gate-ResBlock

## Experimental Results

<img src = "/figs/access_table1.PNG" width="900">
<img src = "/figs/access_table2.PNG" width="450">

### Gaussian Denoising Results
<img src = "/figs/figure_gaussian_gray.PNG" width="900">
The comparison of gray channel denoisers. 

<img src = "/figs/figure_gaussian_color.PNG" width="900">
The comparison of color channel denoisers. 

### Real Noise Denoising Results
<img src = "/figs/figure_real.PNG" width="900">
The comparison of gray channel denoisers. 

## Citation
```
@article{kim2019adaptively,
  title={Adaptively Tuning a Convolutional Neural Network by Gate Process for Image Denoising},
  author={Kim, Yoonsik and Soh, Jae Woong and Cho, Nam Ik},
  journal={IEEE Access},
  volume={7},
  pages={63447--63456},
  year={2019},
  publisher={IEEE}
}
```

