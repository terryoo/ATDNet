import time
from six.moves import xrange
from utils import *
import tensorflow.contrib.slim as slim
from glob import glob
import scipy.misc as ms
from scipy.misc import *

def ATDNet(input, feature_size,noise_sigma, output_channels=1):
    with tf.variable_scope('Generating_CNN'):
        output2 = tf.layers.dense(noise_sigma, 64, use_bias=True, name='dense1')
        generated_kernel = tf.layers.dense(output2, feature_size, use_bias=True, name='dense2')

    with tf.variable_scope('Generating_EDSR'):
        scaling_factor = 0.1
        with tf.variable_scope('block1'):
            x = slim.conv2d(input, feature_size, [3, 3])
            # Store the output of the first convolution to add later
            conv_1 = x
        for layers in xrange(2, 32 + 1):
            with tf.variable_scope('block%d' % layers):
                x = Gate_ResBlock(x, generated_kernel, feature_size, scale=scaling_factor)
        x = slim.conv2d(x, feature_size, [3, 3])
        x += conv_1
        x = slim.conv2d(x, output_channels, [3, 3])
    return x

def Gate_ResBlock(x,generated_kernel,channels=128,kernel_size=[3,3],scale=1):
    tmp = slim.conv2d(x, channels, kernel_size, activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, channels, kernel_size, activation_fn=None)
    with tf.variable_scope('Split_Convolution'):
        gated_results = tf.multiply(tmp,tf.nn.sigmoid(generated_kernel),name = 'multiply')
    tmp = scale * tf.nn.tanh(gated_results)
    return x + tmp

class denoiser(object):

    def __init__(self, sess, input_c_dim=1):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.feature_size = 128
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.noise_matrix = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='noise_matrix')
        self.sigma_matrix = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                           name='sigma_matrix')
        self.X = self.Y_ + self.noise_matrix
        self.Y = ATDNet(self.X, self.feature_size,self.sigma_matrix)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")


    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(self.sess, full_path)
            return True
        else:
            return False

    def test(self, test_files, noise_sigma, model_sigma, ckpt_dir, save_dir):
        """Test DnCNN"""

        # init variables
        tf.initialize_all_variables().run()
        load_model_status= self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")

        test_dataset =  sorted(glob(test_files+'*.png'))
        psnr_sum = 0

        for idx in range(len(test_dataset)):
            file_name = test_dataset[idx].split('/')[-1]
            clean_image = load_images(test_dataset[idx]).astype(np.float32) / 255.0
            noises = np.random.normal(0, noise_sigma / 255, [1, clean_image.shape[1], clean_image.shape[2], 1])
            applied_sigma = np.zeros((1, 1, 1, 1))
            applied_sigma[0, 0, 0, 0] = model_sigma / 255
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image,
                                                                       self.noise_matrix: noises,
                                                                       self.sigma_matrix: applied_sigma})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            psnr_sum += psnr
            print("img%d PSNR: %.2f" % (idx, psnr))

            full_ave_folder = save_dir + '/' + file_name
            ms.imsave(full_ave_folder, outputimage[0, :, :, 0])
        avg_psnr = psnr_sum / len(test_dataset)
        print("--- Test ---- %3d Average PSNR %.2f  ---" % (noise_sigma, avg_psnr))









