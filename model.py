import tensorflow as tf

def lrelu(x, rate=0.1):
    # return tf.nn.relu(x)
    return tf.maximum(tf.minimum(x * rate, 0), x)


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

def generator_c64(z, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_bn_relu(z, 1024)
        fc = fc_bn_relu(fc, 4*4*256)
        conv = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 4, 4, 256]))
        conv = conv2d_t_bn_relu(conv, 192, 4, 2)
        conv = conv2d_t_relu(conv, 128, 4, 2)
        conv = conv2d_t_relu(conv, 128, 4, 1)
        conv = conv2d_t_relu(conv, 64, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv, 3, 4, 2, activation_fn=tf.sigmoid)
        return output


def discriminator_c64(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv = conv2d_lrelu(x, 64, 4, 2)
        conv = conv2d_lrelu(conv, 128, 4, 2)
        conv = conv2d_lrelu(conv, 192, 4, 2)
        conv = conv2d_lrelu(conv, 256, 4, 2)
        fc = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
        fc = fc_lrelu(fc, 1024)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc
