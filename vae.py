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
parser.add_argument('--run', type=int, default=0, help='Index of the run')
parser.add_argument('--architecture', type=str, default='conv')
args = parser.parse_args()

# Hypothesis: optimization gets stuck in local minimum and do not differentiate between the different x
batch_size = 100
data_root = '/home/ubuntu/data/dots_small'
log_root = '/home/ubuntu/data/logc'
# root_path = '/data/dots'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = '%s/vae/model=%s-%s-zdim=%d-lr=%.2f-beta=%.2f-run=%d' % \
       (args.dataset, 'gaussian', args.architecture, args.z_dim, args.lr, args.beta, args.run)
if 'disjoint_count' in args.dataset:
    splited = args.dataset.split('_')
    db_path = []
    for item in splited[2:]:
        db_path.append(os.path.join(data_root, 'disjoint_count_%s' % item))
elif 'combi' in args.dataset:
    splited = args.dataset.split('_')
    db_path = []
    for item in splited[1:]:
        db_path.append(os.path.join(data_root, '%s_%s' % (splited[0], item)))
else:
    db_path = [os.path.join(data_root, args.dataset)]
dataset = DotsDataset(db_path=db_path)

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


idx = 0
start_time = time.time()
while True:
    bx = dataset.next_batch(batch_size)
    _, nll, zkl = sess.run([trainer, loss_elbo, loss_zkl], feed_dict={train_x: bx, kl_anneal: 1 - math.exp(-idx / 10000.0)})

    if idx % 100 == 0:
        summary_val = sess.run(train_summary, feed_dict={train_x: bx, kl_anneal: 1 - math.exp(-idx / 10000.0)})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 1000 == 0:
        summary_val = sess.run(eval_summary, feed_dict={train_x: bx, gen_z: sample_z(100, args.z_dim, 'gaussian')})
        summary_writer.add_summary(summary_val, idx)

    if idx % 10000 == 0:
        bxg_list, bx_list = [], []
        for rep in range(10):
            bxg_list.append(sess.run(gen_x, feed_dict={gen_z: sample_z(1024, args.z_dim, 'gaussian')}))
            bx_list.append(dataset.next_batch(1024))
        bxg = np.concatenate(bxg_list, axis=0)
        bx = np.concatenate(bx_list, axis=0)
        np.savez(os.path.join(model_path, 'samples%d.npz' % (idx // 10000)), g=bxg, x=bx)

    if idx % 10000 == 0:
        saver.save(sess, os.path.join(model_path, "model.ckpt"))

    idx += 1