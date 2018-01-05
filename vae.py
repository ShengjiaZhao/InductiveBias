import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math, os
from tensorflow.examples.tutorials.mnist import input_data
from eval_semi_supervised import *
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--reg_type', type=str, default='mmd', help='Type of regularization')
parser.add_argument('--use_reconstruction', dest='use_reconstruction', action='store_true')
parser.add_argument('-g', '--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('-m', '--z_dim', type=int, default=10, help='Dimension of Latent Code')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.use_reconstruction:
    run_name = 'vae_r_%s_z%d' % (args.reg_type, args.z_dim)
else:
    run_name = 'vae_%s_z%d' % (args.reg_type, args.z_dim)


def make_model_path(name):
    log_path = os.path.join('log', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    fig_path = "%s/fig" % log_path
    os.makedirs(fig_path)
    return log_path, fig_path
log_path, fig_path = make_model_path(run_name)


# Define some handy network layers
def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    conv = lrelu(conv)
    return conv


def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv


def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = lrelu(fc)
    return fc


def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc


# Encoder and decoder use the DC-GAN architecture
def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        return tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)


def decoder(z, reuse=False):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_relu(z, 1024)
        fc2 = fc_relu(fc1, 7*7*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        conv1 = conv2d_t_relu(fc2, 64, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.sigmoid)
        return output


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples):
    cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


mnist = input_data.read_data_sets('mnist_data')

# Build the computation graph for training
train_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
train_z = encoder(train_x, args.z_dim)
train_xr = decoder(train_z)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, args.z_dim])
gen_x = decoder(gen_z, reuse=True)

# Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
true_samples = tf.random_normal(tf.stack([200, args.z_dim]))
loss_mmd = compute_mmd(true_samples, train_z)
loss_nll_per_sample = tf.reduce_mean(tf.square(train_xr - train_x), axis=(1, 2, 3))
loss_nll = tf.reduce_mean(loss_nll_per_sample)
if args.use_reconstruction:
    loss = loss_nll + loss_mmd
else:
    loss = loss_mmd
trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)

train_summary = tf.summary.merge([
    tf.summary.scalar('nll', loss_nll),
    tf.summary.scalar('mmd', loss_mmd),
    tf.summary.scalar('loss', loss)
])
semi100_ph = tf.placeholder(tf.float32)
semi1000_ph = tf.placeholder(tf.float32)
eval_summary = tf.summary.merge([
    tf.summary.scalar('semi-supervised-100', semi100_ph),
    tf.summary.scalar('semi-supervised-1000', semi1000_ph),
])
summary_writer = tf.summary.FileWriter(log_path)

batch_size = 200
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())

# Start training
for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape(-1, 28, 28, 1)
    _, nll, mmd = sess.run([trainer, loss_nll, loss_mmd], feed_dict={train_x: batch_x})
    if i % 10 == 0:
        summary_writer.add_summary(sess.run(train_summary, feed_dict={train_x: batch_x}), i)
    if i % 100 == 0:
        print("Negative log likelihood is %f, mmd loss is %f" % (nll, mmd))
    if i % 500 == 0:
        semi1000, semi100 = semi_supervised_eval(train_x, train_z, sess=sess, mnist=mnist)
        summary_writer.add_summary(sess.run(eval_summary,
                                            feed_dict={semi100_ph:semi100, semi1000_ph:semi1000}), i)
    if False:
        samples = sess.run(gen_x, feed_dict={gen_z: np.random.normal(size=(100, args.z_dim))})
        plt.imshow(convert_to_display(samples), cmap='Greys_r')
        plt.show()
        plt.pause(0.001)
