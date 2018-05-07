from abstract_network import *
from dataset import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=10, help='z dimension')
parser.add_argument('-d', '--dataset', type=str, default='size', help='mnist or random')
parser.add_argument('-lr', '--lr', type=float, default=-3.0, help='learning rate')
parser.add_argument('--beta', type=float, default=1.0, help='Coefficient of KL(q(z|x)||p(z))')
args = parser.parse_args()

# Hypothesis: optimization gets stuck in local minimum and do not differentiate between the different x
batch_size = 100
# root_path = '/home/ubuntu/data/dots_small'
root_path = '/data/dots'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = '%s/vae/model=%s-zdim=%d-lr=%.2f-beta=%.2f' % (args.dataset, 'gaussian', args.z_dim, args.lr, args.beta)
dataset = DotsDataset(db_path=os.path.join(root_path, args.dataset))

encoder = encoder_gaussian_c64
generator = generator_c64

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
nll_per_sample = 64 * tf.reduce_sum(tf.square(train_x - train_xr) + 0.5 * tf.abs(train_x - train_xr), axis=(1, 2, 3))
loss_nll = tf.reduce_mean(nll_per_sample)

kl_anneal = tf.placeholder(tf.float32)
loss_elbo = loss_nll + loss_zkl * args.beta * kl_anneal
trainer = tf.train.AdamOptimizer(10 ** args.lr, beta1=0.5, beta2=0.9).minimize(loss_elbo)

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

model_path = "log2/%s" % name
make_model_path(model_path)
# logger = open(os.path.join(model_path, 'result.txt'), 'w')
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
summary_writer = tf.summary.FileWriter(model_path)
sess.run(tf.global_variables_initializer())

idx = 0
start_time = time.time()
while True:
    bx = dataset.next_batch(batch_size)
    _, nll, zkl = sess.run([trainer, loss_elbo, loss_zkl], feed_dict={train_x: bx, kl_anneal: 1 - math.exp(-idx / 10000.0)})

    if idx % 100 == 0:
        summary_val = sess.run(train_summary, feed_dict={train_x: bx, kl_anneal: 1 - math.exp(-idx / 10000.0)})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 500 == 0:
        summary_val = sess.run(eval_summary, feed_dict={train_x: bx, gen_z: sample_z(100, args.z_dim, 'gaussian')})
        summary_writer.add_summary(summary_val, idx)

        bxg = sess.run(g, feed_dict={z: sample_z(512, args.z_dim, 'gaussian')})
        np.save(os.path.join(model_path, 'samples%d.npy' % (idx // 10000)), bxg)

        if 'size' in args.dataset:
            bx = dataset.next_batch(512)
            bx_size = dataset.eval_size(bx)
            bxg_size = dataset.eval_size(bxg)
            summary_val = sess.run(hist_summary, feed_dict={train_size_ph: bx_size, gen_size_ph: bxg_size})
            summary_writer.add_summary(summary_val, idx)

    idx += 1