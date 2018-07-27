from abstract_network import *
from dataset import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=20, help='z dimension')
parser.add_argument('-d', '--dataset', type=str, default='size')
parser.add_argument('-lr', '--lr', type=float, default=-4.0, help='learning rate')
parser.add_argument('--beta', type=float, default=1.0, help='Coefficient of KL(q(z|x)||p(z))')
parser.add_argument('--run', type=int, default=0, help='Index of the run')
parser.add_argument('--architecture', type=str, default='fc')
args = parser.parse_args()

batch_size = 100
log_root = '/home/ubuntu/data/logr'
# data_root = '/data/dots'
# log_root = 'log'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = '%s/vae/model=%s-%s-zdim=%d-lr=%.2f-beta=%.2f-run=%d' % \
       (args.dataset, 'gaussian', args.architecture, args.z_dim, args.lr, args.beta, args.run)

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

if 'conv' in args.architecture:
    encoder = encoder_gaussian_c64
    generator = generator_c64
else:
    encoder = encoder_c64fc
    generator = generator_c64fc

# Build the computation graph for training
train_x = tf.placeholder(tf.float32, shape=[None] + dataset.data_dims)
train_zdist, train_zsample = encoder(train_x, args.z_dim)
# ELBO loss divided by input dimensions
zkl_per_sample = tf.reduce_sum(-tf.log(train_zdist[1]) + 0.5 * tf.square(train_zdist[1]) +
                               0.5 * tf.square(train_zdist[0]) - 0.5, axis=1)
loss_zkl = tf.reduce_mean(zkl_per_sample)
train_xr = generator(train_zsample)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, args.z_dim])
gen_x = generator(gen_z, reuse=True)

# Negative log likelihood per dimension
nll_per_sample = tf.reduce_sum(tf.square(train_x - train_xr) + 0.5 * tf.abs(train_x - train_xr), axis=(1, 2, 3))
loss_nll = tf.reduce_mean(nll_per_sample)

kl_anneal = tf.placeholder(tf.float32)
loss_elbo = loss_nll + loss_zkl * args.beta * kl_anneal
trainer = tf.train.AdamOptimizer(10 ** args.lr, beta1=0.5, beta2=0.9).minimize(loss_elbo)
saver = tf.train.Saver(var_list=[var for var in tf.global_variables() if 'i_net' in var.name or 'g_net' in var.name])

train_summary = tf.summary.merge([
    tf.summary.scalar('loss_zkl', loss_zkl),
    tf.summary.scalar('loss_nll', loss_nll),
    tf.summary.scalar('loss_elbo', loss_elbo),
])

eval_summary = tf.summary.merge([
    create_display(tf.reshape(gen_x, [batch_size]+dataset.data_dims), 'samples'),
    create_display(tf.reshape(train_xr, [batch_size]+dataset.data_dims), 'reconstructions'),
    create_display(tf.reshape(train_x, [batch_size]+dataset.data_dims), 'train_samples')
])

train_size_ph = tf.placeholder(tf.float32, shape=[None])
gen_size_ph = tf.placeholder(tf.float32, shape=[None])
hist_summary = tf.summary.merge([
    tf.summary.histogram('train', train_size_ph),
    tf.summary.histogram('gen', gen_size_ph),
])

model_path = os.path.join(log_root, name)
make_model_path(model_path)
# logger = open(os.path.join(model_path, 'result.txt'), 'w')
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
summary_writer = tf.summary.FileWriter(model_path)
sess.run(tf.global_variables_initializer())


start_time = time.time()
for idx in range(1, 400001):
    bx = dataset.next_batch(batch_size)
    _, nll, zkl = sess.run([trainer, loss_elbo, loss_zkl], feed_dict={train_x: bx, kl_anneal: 1 - math.exp(-idx / 50000.0)})

    if idx % 100 == 0:
        summary_val = sess.run(train_summary, feed_dict={train_x: bx, kl_anneal: 1 - math.exp(-idx / 50000.0)})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 2000 == 0:
        summary_val = sess.run(eval_summary, feed_dict={train_x: bx, gen_z: sample_z(100, args.z_dim, 'gaussian')})
        summary_writer.add_summary(summary_val, idx)

    if idx % 10000 == 0:
        bxg_list, bx_list = [], []
        for rep in range(10):
            bxg_list.append(sess.run(gen_x, feed_dict={gen_z: sample_z(1024, args.z_dim, 'gaussian')}))
            bx_list.append(dataset.next_batch(1024))
        bxg = np.concatenate(bxg_list, axis=0)
        bx = np.concatenate(bx_list, axis=0)
        np.savez(os.path.join(model_path, 'samples%d.npz' % (1 + idx // 50000)), g=bxg, x=bx)

    if idx % 10000 == 0:
        saver.save(sess, os.path.join(model_path, "model.ckpt"))

