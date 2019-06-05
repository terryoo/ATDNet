import time
from six.moves import xrange
from utils import *
import tensorflow.contrib.slim as slim
from glob import glob
import scipy.misc as ms
from scipy.misc import *


def ATDNet(input, feature_size,noise_sigma, output_channels=1):
    with tf.variable_scope('Generating_CNN'):
        x = tf.layers.conv2d(noise_sigma, 64, [1, 1], activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',
                             name='dense1')
        x = tf.layers.conv2d(x, 128, [1, 1], activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',
                             name='dense2')
        x = tf.layers.conv2d(x, 128, [1, 1], activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',
                             name='dense3')
        generated_kernel = tf.layers.conv2d(x, 128, [1, 1],
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',
                             name='dense4')

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

def Estimation_block2(input):
    # batch x 128 x 128 x 1
    with tf.variable_scope('Estimation_block2'):
        output = tf.layers.conv2d(input, 32, 3, padding='same', activation=tf.nn.relu, name='conv1')
        output = tf.layers.conv2d(output, 32, 3, padding='same', activation=tf.nn.relu, name='conv2')
        output = tf.nn.max_pool(output,[1,2,2,1],[1,2,2,1],padding='VALID',name='maxpool1') # batch x 40 x 40 x 32

        output = tf.layers.conv2d(output, 64, 3, padding='same', activation=tf.nn.relu, name='conv3')
        output = tf.layers.conv2d(output, 64, 3, padding='same', activation=tf.nn.relu, name='conv4')
        output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1],padding='VALID', name='maxpool2') # batch x 20 x 20 x 64

        output = tf.layers.conv2d(output, 128, 3, padding='same', activation=tf.nn.relu, name='conv5')
        output = tf.nn.avg_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='avgpool') # batch x 10 x 10 x 128

        output = tf.layers.conv2d(output, 1, 1, padding='same', activation=tf.nn.relu, name='conv6')
        h_sigma = tf.reduce_mean(tf.reduce_mean(output,axis=1,keep_dims=True),axis=2,keep_dims=True,name='h_sigma')  # batch x 1 x 1 x 1

    return h_sigma,output

class denoiser(object):

    def __init__(self, sess, input_c_dim = 1):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.feature_size = 128

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='gt')
        self.noise_matrix = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                           name='noise_matrix')

        self.X = self.Y_ + self.noise_matrix  # noisy images
        self.h_sigma, self.output = Estimation_block2(self.X)
        self.h_sigma = tf.round(self.h_sigma * 255) / 255.
        self.Y = ATDNet(self.X ,self.feature_size, self.h_sigma, self.input_c_dim)
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
            return False,

    def load_pre(self, checkpoint_dir,var_name):
        print("[*] Reading checkpoint...")
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=var_name)

        saver = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(self.sess, full_path)
            return True
        else:
            return False

    def test(self, test_files, noise_sigma, ckpt_dir, save_dir):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        # assert len(test_files) != 0, 'No testing data!'
        ckpt_dir2 = './access_model/ESTIMATION'
        load_model_status = self.load_pre(ckpt_dir, 'Generating_CNN')
        load_model_status2= self.load_pre(ckpt_dir2, 'Estimation_block2')
        load_model_status3 = self.load_pre(ckpt_dir, 'Generating_EDSR')
        assert load_model_status3 == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")

        test_dataset = sorted(glob(test_files + '*.png'))
        psnr_sum = 0

        for idx in range(len(test_dataset)):
            file_name = test_dataset[idx].split('/')[-1]
            clean_image = load_images(test_dataset[idx]).astype(np.float32) / 255.0
            noises = np.random.normal(0, noise_sigma / 255, [1, clean_image.shape[1], clean_image.shape[2], 1])
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image,
                                                                       self.noise_matrix: noises})
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


