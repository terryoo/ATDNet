import argparse
from glob import glob
import numpy as np
import os
import tensorflow as tf
from model_atdnet import denoiser
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--noise_sigma', dest='noise_sigma', type=int, default=25, help='noise level')
parser.add_argument('--model_sigma', dest='model_sigma', type=int, default=25, help='model or applied sigma level')
parser.add_argument('--phase', dest='phase', default='test', help='test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./access_model/ATDNETW', help='models are saved here') # option ./ATDNET or ./ATDNETW
parser.add_argument('--test_set', dest='test_set', default='./testset/BSD68/', help='dataset for testing')
parser.add_argument('--result_dir', dest='result_dir', default='./results/', help='test sample are saved here')
args = parser.parse_args()


def denoiser_test(denoiser):
    denoiser.test(args.test_set, args.noise_sigma, args.model_sigma, ckpt_dir=args.ckpt_dir, save_dir=args.result_dir)

def main(_):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            denoiser_test(model)

    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, sigma=args.sigma)
            denoiser_test(model)

if __name__ == '__main__':
    tf.app.run()
