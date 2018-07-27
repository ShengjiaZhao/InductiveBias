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
parser.add_argument('--dest', type=str, default='/data/dots/')
parser.add_argument('--noisy', type=bool, default='False')
parser.add_argument('--type', type=str, default='combi_46555', help='color or location')
args = parser.parse_args()

args.dest = os.path.join(args.dest, args.type)

params = args.type.split('_')[1]

images = []
start_time = time.time()
for i in range(args.bs):
    new_img = gen_combi(params)
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

