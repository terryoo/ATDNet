import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow.contrib.slim as slim

def resBlock(x,channels=64,kernel_size=[3,3],scale=1):
	tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
	tmp = tf.nn.relu(tmp)
	tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
	tmp *= scale
	return x + tmp

def resBlock(x,channels=64,kernel_size=[3,3],scale=1):
	tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
	tmp = tf.nn.relu(tmp)
	tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
	tmp *= scale
	return x + tmp



def resBlock3D(x,channels=64,kernel_size=[3,3,3],scale=1):
    tmp = tf.layers.conv3d(x,channels,kernel_size,padding='same',name='conv1')
    tmp = tf.nn.relu(tmp)
    tmp = tf.layers.conv3d(tmp, channels, kernel_size, padding='same', name='conv2')
    tmp *= scale
    return x + tmp

def resBlock3D_lr(x,channels=64,kernel_size=[3,3,3],scale=1):
    tmp = tf.layers.conv3d(x,channels,kernel_size,padding='same',name='conv1')
    tmp = tf.nn.leaky_relu(tmp,0.2)
    tmp = tf.layers.conv3d(tmp, channels, kernel_size, padding='same', name='conv2')
    tmp *= scale
    return x + tmp



def resBlock_generating(x,generated_kernel,batch_size=16,channels=64,kernel_size=[3,3],scale=1):
    tmp = slim.conv2d(x, channels, kernel_size, activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, channels, kernel_size, activation_fn=None)
    with tf.variable_scope('Split_Convolution'):
        result_list = []
        for i in range(batch_size):
            output_image = tf.expand_dims(tmp[i, :, :, :], axis=0)
            generated_piece = generated_kernel[i, :, :, :]
            filtered_output = tf.multiply(output_image,tf.nn.tanh(generated_piece))
            result_list.append(filtered_output)
        output_list = tf.stack(result_list)
        output_list = tf.squeeze(output_list, axis=1)
    tmp *= scale
    return x + tmp

def resBlock_generating2(x,generated_kernel,batch_size=16,channels=128,kernel_size=[3,3],scale=1):
    tmp = slim.conv2d(x, channels, kernel_size, activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, channels, kernel_size, activation_fn=None)
    with tf.variable_scope('Split_Convolution'):
        gated_results = tf.multiply(tmp,tf.nn.sigmoid(generated_kernel),name = 'multiply')
    tmp = scale * tf.nn.tanh(gated_results)
    return x + tmp


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


class train_data():
    def __init__(self, filepath='./data/image_clean_pat.npy'):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath='./data/image_clean_pat.npy'):
    return train_data(filepath=filepath)


def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data
def load_images_2d(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data

def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')

def save_images_color(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('RGB')
    im.save(filepath, 'png')

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))

