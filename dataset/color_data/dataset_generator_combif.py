# Only works with python 2
import matplotlib
matplotlib.use('Agg')
from utils import *
import time
import os
import numpy as np
import argparse
from matplotlib import colors as pltcolor

parser = argparse.ArgumentParser()
parser.add_argument('--bn', type=int, default=0)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--dest', type=str, default='/home/ubuntu/data/dots/')
parser.add_argument('--noisy', type=bool, default='False')
parser.add_argument('--configs', type=int, default=100, help='color or location')
parser.add_argument('--index', type=int, default=0, help='index of dataset')
args = parser.parse_args()

if args.noisy:
    args.dest = os.path.join(args.dest, 'combirn_%d_%d' % (args.configs, args.index))
else:
    args.dest = os.path.join(args.dest, 'combir_%d_%d' % (args.configs, args.index))

configs = []
size_list = [4, 5, 6, 7]
locx_list = [2, 4, 5, 6, 8]
locy_list = [2, 4, 5, 6, 8]
color_list = [1, 2, 3, 5, 9]

np.random.seed(args.index)
for config in range(args.configs):
    param = None
    while param is None or param in configs:
        size = np.random.choice(size_list)
        locx = np.random.choice(locx_list)
        locy = np.random.choice(locy_list)
        color = np.random.choice(color_list)
        param = '4%d%d%d%d' % (size, locx, locy, color)
    configs.append(param)
print(configs)

images = []
start_time = time.time()
for i in range(args.bs):
    new_img = gen_combi(np.random.choice(configs))

    if args.noisy:
        new_img += np.random.normal(loc=0, scale=0.03, size=new_img.shape)
        new_img = 1.0 - np.abs(1.0 - new_img)
        new_img = np.abs(new_img)
    images.append(new_img)
    if (i+1) % 1000 == 0:
        print("Generating %d-th image, time used: %f" % (i+1, time.time() - start_time))

batch_img = np.stack(images, axis=0)
if not os.path.isdir(args.dest):
    os.makedirs(args.dest)
np.savez(os.path.join(args.dest, 'batch%d.npz' % args.bn), images=batch_img)

