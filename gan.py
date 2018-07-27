from dataset import *
from abstract_network import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=100, help='z dimension')
parser.add_argument('-m', '--model', type=str, default='bernoulli_wgan', help='gaussian or discrete')
parser.add_argument('-d', '--dataset', type=str, default='color_band_4_3', help='mnist or random')
parser.add_argument('-lr', '--lr', type=float, default=-4, help='learning rate')
parser.add_argument('-r', '--repeat', type=int, default=3, help='Number of times to train discriminator each time generator is trained')
args = parser.parse_args()

batch_size = 100
data_root = '/home/ubuntu/data/dots'
log_root = '/home/ubuntu/data/logc'
# data_root = '/data/dots'
# log_root = 'log'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = '%s/gan/model=%s-zdim=%d-lr=%.2f-rep=%d' % (args.dataset, args.model, args.z_dim, args.lr, args.repeat)
if 'disjoint_count' in args.dataset:
    splited = args.dataset.split('_')
    db_path = []
    for item in splited[2:]:
        db_path.append(os.path.join(data_root, 'disjoint_count_%s' % item))
elif 'color_band' in args.dataset:
    splited = args.dataset.split('_')
    db_path = []
    for item in splited[3:]:
        db_path.append(os.path.join(data_root, 'color_band_%s_%s' % (splited[2], item)))
elif 'color_random' in args.dataset:
    splited = args.dataset.split('_')
    db_path = []
    for item in splited[3:]:
        db_path.append(os.path.join(data_root, 'color_random_%s_%s' % (splited[2], item)))
elif 'combi' in args.dataset:
    splited = args.dataset.split('_')
    db_path = []
    for item in splited[1:]:
        db_path.append(os.path.join(data_root, '%s_%s' % (splited[0], item)))
else:
    db_path = [os.path.join(data_root, args.dataset)]
dataset = DotsDataset(db_path=db_path)

z = tf.placeholder(tf.float32, [None, args.z_dim])
x = tf.placeholder(tf.float32, [None] + dataset.data_dims)
generator = generator_c64
discriminator = discriminator_c64

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
idx = 1
while True:
    bx = dataset.next_batch(batch_size)
    bz = sample_z(batch_size, args.z_dim, args.model)
    for i in range(args.repeat - 1):
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
            bxg = sess.run(g, feed_dict={z: sample_z(1024, args.z_dim, 'gaussian')})
            bx = dataset.next_batch(1024)
        np.savez(os.path.join(model_path, 'samples%d.npz' % (idx // 10000)), g=bxg, x=bx)

        eval_func = None
        if 'color_gap' in args.dataset:
            eval_func = dataset.eval_color_gap
        elif 'disjoint_count' in args.dataset:
            eval_func = dataset.eval_count
        elif 'size' in args.dataset:
            eval_func = dataset.eval_size
        elif 'color_count' in args.dataset or 'color_random' in args.dataset or 'color_band' in args.dataset:
            eval_func = dataset.eval_color_proportion
        if eval_func is not None:
            bx_cnt = eval_func(bx)
            bxg_cnt = eval_func(bxg)
            summary_val = sess.run(hist_summary, feed_dict={train_size_ph: bx_cnt, gen_size_ph: bxg_cnt})
            summary_writer.add_summary(summary_val, idx)

    if idx % 10000 == 0:
        cur_lr *= 0.5

    if idx % 10000 == 0:
        saver.save(sess, os.path.join(model_path, "model.ckpt"))

    idx += 1
    if idx == 100000:
        break