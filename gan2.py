from dataset import *
from abstract_network import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=100, help='z dimension')
parser.add_argument('-m', '--model', type=str, default='bernoulli_wgan', help='gaussian or discrete')
parser.add_argument('-d', '--dataset', type=str, default='crand_100', help='mnist or random')
parser.add_argument('-lr', '--lr', type=float, default=-4, help='learning rate')
parser.add_argument('-r', '--repeat', type=int, default=2, help='Number of times to train discriminator each time generator is trained')
parser.add_argument('--run', type=int, default=0, help='Index of the run')
parser.add_argument('--architecture', type=str, default='fc')
args = parser.parse_args()

batch_size = 100
log_root = '/home/ubuntu/data/logr'
# data_root = '/data/dots'
# log_root = 'log'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = '%s/gan/model=%s-%s-zdim=%d-lr=%.2f-rep=%d-run=%d' % \
       (args.dataset, args.model, args.architecture, args.z_dim, args.lr, args.repeat, args.run)

splited = args.dataset.split('_')
num_params = int(splited[1])
fixed_dim = -1

if len(splited) > 2:
    fixed_dim = int(splited[2])
    fixed_options = [int(item) for item in list(splited[3])]
    cur_option = 0

params = []
size_list = [1, 3, 5, 7, 9]
locx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
locy_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
color_list = [1, 3, 5, 7, 9]

np.random.seed(args.run)
for i in range(num_params):
    param = None
    while param is None or param in params:
        if fixed_dim != 0:
            size = np.random.choice(size_list)
        else:
            size = fixed_options[cur_option]
        if fixed_dim != 1:
            locx = np.random.choice(locx_list)
        else:
            locx = fixed_options[cur_option]
        if fixed_dim != 2:
            locy = np.random.choice(locy_list)
        else:
            locy = fixed_options[cur_option]
        if fixed_dim != 3:
            color = np.random.choice(color_list)
        else:
            color = fixed_options[cur_option]
        param = '4%d%d%d%d' % (size, locx, locy, color)
    if fixed_dim >= 0:
        cur_option = (cur_option + 1) % len(fixed_options)
    params.append(param)

dataset = DotsDataset2(params=params)

z = tf.placeholder(tf.float32, [None, args.z_dim])
x = tf.placeholder(tf.float32, [None] + dataset.data_dims)

if 'conv' in args.architecture:
    discriminator = discriminator_c64
    generator = generator_c64
elif 'small' in args.architecture:
    discriminator = discriminator_c64s
    generator = generator_c64s
else:
    discriminator = discriminator_c64fc
    generator = generator_c64fc


g = generator(z)
d = discriminator(x)
d_ = discriminator(g, reuse=True)

# Gradient penalty
epsilon = tf.random_uniform([], 0.0, 1.0)
x_hat = epsilon * x + (1 - epsilon) * g
d_hat = discriminator(x_hat, reuse=True)

ddx = tf.gradients(d_hat, x_hat)[0]
ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=(1, 2, 3)))
d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

if 'dcgan' in args.model:
    d_loss_x = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d, labels=tf.ones_like(d)))
    d_loss_g = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_, labels=tf.zeros_like(d_)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_, labels=tf.ones_like(d_)))
    d_loss = d_loss_x + d_loss_g
    d_confusion = tf.reduce_mean(d_) - tf.reduce_mean(d)
else:
    d_loss_x = -tf.reduce_mean(d)
    d_loss_g = tf.reduce_mean(d_)
    d_loss = d_loss_x + d_loss_g + d_grad_loss
    d_confusion = tf.reduce_mean(d) - tf.reduce_mean(d_)
    g_loss = -tf.reduce_mean(d_)

d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
g_vars = [var for var in tf.global_variables() if 'g_net' in var.name]
lr = tf.placeholder(tf.float32)
cur_lr = 10 ** args.lr
d_train = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_train = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)
saver = tf.train.Saver(var_list=[var for var in tf.global_variables() if 'd_net' in var.name or 'g_net' in var.name])

train_summary = tf.summary.merge([
    tf.summary.scalar('g_loss', g_loss),
    tf.summary.scalar('d_loss', d_loss),
    tf.summary.scalar('confusion', d_confusion),
    tf.summary.scalar('d_loss_g', d_loss_g),
])

eval_summary = tf.summary.merge([
    create_display(tf.reshape(g, [100]+dataset.data_dims), 'samples'),
    create_display(tf.reshape(x, [100]+dataset.data_dims), 'train_samples')
])

model_path = os.path.join(log_root, name)
make_model_path(model_path)

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
summary_writer = tf.summary.FileWriter(model_path)
sess.run(tf.global_variables_initializer())

start_time = time.time()

for idx in range(1, 200001):
    bx = dataset.next_batch(batch_size)
    bz = sample_z(batch_size, args.z_dim, args.model)
    for _ in range(args.repeat - 1):
        sess.run(d_train, feed_dict={x: bx, z: bz, lr: cur_lr})
    sess.run([d_train, g_train], feed_dict={x: bx, z: bz, lr: cur_lr})

    if idx % 100 == 0:
        summary_val = sess.run(train_summary, feed_dict={x: bx, z: bz, lr: cur_lr})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 2000 == 0:
        summary_val = sess.run(eval_summary, feed_dict={x: bx, z: sample_z(100, args.z_dim, args.model)})
        summary_writer.add_summary(summary_val, idx)

    if idx % 10000 == 0:
        bxg_list, bx_list = [], []
        for rep in range(10):
            bxg_list.append(sess.run(g, feed_dict={z: sample_z(1024, args.z_dim, 'gaussian')}))
            bx_list.append(dataset.next_batch(1024))
        bxg = np.concatenate(bxg_list, axis=0)
        bx = np.concatenate(bx_list, axis=0)
        np.savez(os.path.join(model_path, 'samples%d.npz' % (1 + idx // 50000)), g=bxg, x=bx)

    if idx % 10000 == 0:
        saver.save(sess, os.path.join(model_path, "model.ckpt"))
    if idx % 20000 == 0:
        cur_lr *= 0.5
