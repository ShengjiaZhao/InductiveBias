import tensorflow as tf
import numpy as np
import math
import glob
import os, sys, shutil, re

def lrelu(x, rate=0.1):
    # return tf.nn.relu(x)
    return tf.maximum(tf.minimum(x * rate, 0), x)

conv2d = tf.contrib.layers.convolution2d
conv2d_t = tf.contrib.layers.convolution2d_transpose
fc_layer = tf.contrib.layers.fully_connected


def make_model_path(model_path):
    import subprocess
    if os.path.isdir(model_path):
        subprocess.call(('rm -rf %s' % model_path).split())
    os.makedirs(model_path)


def conv2d_bn_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = lrelu(conv)
    return conv


def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           activation_fn=tf.identity)
    conv = lrelu(conv)
    return conv


def conv2d_t_bn_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = tf.nn.relu(conv)
    return conv

def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv

def conv2d_t_bn_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    conv = lrelu(conv)
    return conv


def conv2d_t_bn(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv)
    return conv


def fc_bn(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc)
    return fc

def fc_bn_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc)
    fc = lrelu(fc)
    return fc


def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           activation_fn=tf.identity)
    fc = lrelu(fc)
    return fc

def fc_bn_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc)
    fc = tf.nn.relu(fc)
    return fc

def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples, max_samples=100):
    if max_samples > samples.shape[0]:
        max_samples = samples.shape[0]
    cnt, height, width = int(math.floor(math.sqrt(max_samples))), samples.shape[1], samples.shape[2]
    samples = samples[:cnt*cnt]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


# Input a tensor of shape [batch_size, height, width, color_dim], tiles the tensor for display
# color_dim can be 1 or 3. The tensor should be normalized to lie in [0, 1]
def create_display(tensor, name):
    channels, height, width, color_dim = [tensor.get_shape()[i].value for i in range(4)]
    cnt = int(math.floor(math.sqrt(float(channels))))
    if color_dim >= 3:
        tensor = tf.slice(tensor, [0, 0, 0, 0], [cnt * cnt, -1, -1, 3])
        color_dim = 3
    else:
        tensor = tf.slice(tensor, [0, 0, 0, 0], [cnt * cnt, -1, -1, 1])
        color_dim = 1
    tensor = tf.pad(tensor, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    tensor = tf.reshape(tensor, [height + 2, cnt, cnt, width + 2, color_dim])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [1, (height + 2) * cnt, (width + 2) * cnt, color_dim])
    return tf.summary.image(name, tensor, max_outputs=1)


def create_multi_display(tensors, name):
    channels, height, width, color_dim = [tensors[0].get_shape()[i].value for i in range(4)]
    max_columns = 15
    columns = int(math.floor(float(max_columns) / len(tensors)))
    rows = int(math.floor(float(channels) / columns))
    if rows == 0:
        columns = channels
        rows = 1

    for index in range(len(tensors)):
        if color_dim >= 3:
            tensors[index] = tf.slice(tensors[index], [0, 0, 0, 0], [rows * columns, -1, -1, 3])
            color_dim = 3
        else:
            tensors[index] = tf.slice(tensors[index], [0, 0, 0, 0], [rows * columns, -1, -1, 1])
            color_dim = 1
    tensor = tf.stack(tensors)
    tensor = tf.transpose(tensor, [1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [-1, height, width, color_dim])
    tensor = tf.pad(tensor, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    tensor = tf.reshape(tensor, [height + 2, rows, columns * len(tensors), width + 2, color_dim])
    tensor = tf.transpose(tensor, perm=[1, 0, 2, 3, 4])
    tensor = tf.reshape(tensor, [1, (height + 2) * rows, (width + 2) * columns * len(tensors), color_dim])
    return tf.summary.image(name, tensor, max_outputs=1)


class Network:
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = dataset.batch_size

        self.learning_rate_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name="lr_placeholder")

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        # A unique name should be given to each instance of subclasses during initialization
        self.name = "default"

        # These should be updated accordingly
        self.iteration = 0
        self.learning_rate = 0.0
        self.read_only = False

        self.do_generate_samples = False
        self.do_generate_conditional_samples = False
        self.do_generate_manifold_samples = False

    def make_model_path(self):
        if not os.path.isdir("models"):
            os.mkdir("models")
        if not os.path.isdir("models/" + self.name):
            os.mkdir("models/" + self.name)

    def print_network(self):
        self.make_model_path()
        if os.path.isdir("models/" + self.name):
            for f in os.listdir("models/" + self.name):
                if re.search(r"events.out*", f):
                    os.remove(os.path.join("models/" + self.name, f))
        self.writer = tf.summary.FileWriter("models/" + self.name, self.sess.graph)
        self.writer.flush()

    """ Save network, if network file already exists back it up to models/old folder. Only one back up will be created
    for each network """
    def save_network(self):
        if not self.read_only:
            # Saver and Summary ops cannot run in GPU
            with tf.device('/cpu:0'):
                saver = tf.train.Saver()
            self.make_model_path()
            if not os.path.isdir("models/old"):
                os.mkdir("models/old")
            file_name = "models/" + self.name + "/" + self.name + ".ckpt"
            if os.path.isfile(file_name):
                os.rename(file_name, "models/old/" + self.name + ".ckpt")
            saver.save(self.sess, file_name)

    """ Either initialize or load network from file.
    Always run this at end of initialization for every subclass to initialize Variables properly """
    def init_network(self, restart=False):
        self.sess.run(tf.global_variables_initializer())
        if restart:
            return
        file_name = "models/" + self.name + "/" + self.name + ".ckpt"
        if len(glob.glob(file_name + '*')) != 0:
            saver = tf.train.Saver()
            try:
                saver.restore(self.sess, file_name)
            except:
                print("Warning: network load failed, reinitializing all variables", sys.exc_info()[0])
                self.sess.run(tf.global_variables_initializer())
        else:
            print("No checkpoint file found, Initializing model from random")

    """ This function should train on the given batch and return the training loss """
    def train(self, batch_input, batch_target, labels=None):
        return None

    """ This function should take the input and return the reconstructed images """
    def test(self, batch_input, labels=None):
        return None

    """ This function should take the input and return latents states whose activation we would like to visualize"""
    def get_visualization(self, batch_input):
        return None

    """ This function should return a sample generated by the network """
    def generate_samples(self):
        return None

    def latent_feature(self, batch_input):
        return None

    def latent_activation(self, batch_input, layer):
        return None

    @staticmethod
    def fc_layer(input_tensor, output_dim, name="fc"):
        with tf.variable_scope(name):
            weight = tf.get_variable("weight", [input_tensor.get_shape()[1].value, output_dim],
                                     initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / output_dim)))
            bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.add(tf.matmul(input_tensor, weight, name="matmul"), bias, name="bias")

    @staticmethod
    def fc_with_wn(input_tensor, output_dim, name="wnfc"):
        with tf.variable_scope(name):
            weight = tf.get_variable("weight", [input_tensor.get_shape()[1].value, output_dim],
                                     initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / output_dim)))
            bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(0.0))
            ratio = tf.get_variable("ratio", [], initializer=tf.constant_initializer(1.0))
            norm = tf.sqrt(tf.reduce_sum(tf.square(weight)), name="norm")
            normed_weight = tf.mul(ratio, tf.div(weight, norm), name="normalized_weight")
            return tf.add(tf.matmul(input_tensor, normed_weight, name="matmul"), bias, name="bias")

    def stochastic_variable(self, output_dim, name=None):
        return tf.Variable(tf.random_normal([self.dataset.batch_size, output_dim], stddev=1.0, name=name))

    def fc_network(self, input_tensor, output_dim, hidden_dims=(50, 20), name="fc_network"):
        with tf.name_scope(name):
            prev_fc = input_tensor
            for layer_index, dim in enumerate(hidden_dims):
                fc = tf.nn.relu(self.fc_with_wn(prev_fc, output_dim=dim, name="fc_%d" % (layer_index + 1)))
                prev_fc = fc
            mean = self.fc_layer(prev_fc, output_dim=output_dim, name="fc_mean")
            stddiv = 2 * tf.nn.sigmoid(self.fc_layer(prev_fc, output_dim=output_dim, name="fc_stddiv")) + 0.01
            return mean, stddiv

    def fc_shortcut_network(self, input_tensor, output_dim, hidden_dims=(50, 20), name="fc_network"):
        with tf.name_scope(name):
            prev_fc = input_tensor
            for layer_index, dim in enumerate(hidden_dims):
                fc = tf.nn.relu(self.fc_with_wn(prev_fc, output_dim=dim, name="fc_%d" % (layer_index + 1)))
                prev_fc = fc
            mean = self.fc_layer(prev_fc, output_dim=output_dim, name="fc_mean") + \
                   self.fc_layer(input_tensor, output_dim=output_dim, name="mean_shortcut")
            stddiv = 2 * tf.nn.sigmoid(self.fc_layer(prev_fc, output_dim=output_dim, name="fc_stddiv")) + \
                     self.fc_layer(input_tensor, output_dim=output_dim, name="stddiv_shortcut") + 0.01
            return mean, stddiv
