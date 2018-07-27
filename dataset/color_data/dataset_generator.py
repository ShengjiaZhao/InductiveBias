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
parser.add_argument('--noisy', type=bool, default='True')
parser.add_argument('--type', type=str, default='color_band_4_5', help='color or location')
args = parser.parse_args()

args.dest = os.path.join(args.dest, args.type)

if args.type == 'color':
    color_codes = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]).astype(np.float)
    ref = gen_image_color(color_codes, small=True)
    masks = []
    for i in range(6):
        color_code = color_codes[i]
        color_mask = np.logical_and(np.logical_and(
            ref[:, :, 0] == color_code[0]*255, ref[:, :, 1] == color_code[1]*255),
                                    ref[:, :, 2] == color_code[2]*255).astype(np.float)
        masks.append(np.expand_dims(color_mask, axis=-1))
    masks = np.stack(masks, axis=0)
else:
    masks = None

images = []
colors = []
start_time = time.time()
for i in range(args.bs):
    if args.type == 'color':
        random_color = np.random.uniform(0, 0.9, size=(6, 3))
        new_img = gen_image_color(random_color).astype(np.float32) / 255.0
    elif args.type == 'location':
        random_color = np.random.uniform(0, 0.9, size=(1, 3))
        new_img = gen_image_location(random_color).astype(np.float32) / 255.0
    elif 'disjoint_count' in args.type:
        values = args.type.split('_')[2:]
        num_object = int(values[np.random.randint(0, len(values))])
        random_color = np.random.uniform(0, 0.9, size=(12, 3))
        new_img = gen_image_count(random_color, num_object).astype(np.float32) / 255.0
    elif 'overlap_count' in args.type:
        values = args.type.split('_')[2:]
        num_object = int(values[np.random.randint(0, len(values))])
        random_color = np.random.uniform(0, 0.9, size=(12, 3))
        new_img = gen_image_count(random_color, 6, overlap=True).astype(np.float32) / 255.0
    elif 'color_count' in args.type:
        count = int(args.type.split('_')[2])
        color = int(args.type.split('_')[3])
        random_color = np.random.uniform(0, 0.9, size=(12, 3))
        random_color[:color, 0] = np.random.uniform(0.8, 0.9, size=(color,))
        random_color[:color, 1:] = 0.0
        random_color[color:, 0] = 0.0
        new_img = gen_image_count(random_color, count).astype(np.float32) / 255.0
    elif 'size' in args.type:
        gap = int(args.type.split('_')[1])
        random_color = np.random.uniform(0, 0.9, size=(1, 3))

        new_img = gen_image_size(random_color, 1, gap=float(gap) / 100.0).astype(np.float32) / 255.0
    elif 'color_gap' in args.type:  # color_gap
        gap = int(args.type.split('_')[2])
        gap = float(gap) / 100.0
        if np.random.randint(0, 2) == 0:
            color = np.random.uniform(gap / 2, 0.5 - gap / 2)
        else:
            color = np.random.uniform(0.5 + gap / 2, 1.0 - gap / 2)
        random_color = pltcolor.hsv_to_rgb(np.array([[color, 1.0, 1.0]]))
        new_img = gen_image_size(random_color, gap=0.0, num_object=1).astype(np.float32) / 255.0
    elif 'color_random' in args.type:  # color_random
        count = int(args.type.split('_')[2])
        color = float(args.type.split('_')[3]) / 10.0
        random_color = np.random.uniform(0, 0.9, size=(count, 3))
        random_color[0, 0] = np.random.uniform(0.8, 0.9)
        random_color[0, 1:] = 0.0
        random_color[1:, 0] = 0.0
        random_weights = np.random.uniform(0.01, 1, size=count-1)
        random_weights = np.concatenate([[color], (1 - color) * random_weights / np.sum(random_weights)])
        new_img = gen_color_random(random_color, random_weights)
    else: # color_band
        count = int(args.type.split('_')[2])
        color = float(args.type.split('_')[3]) / 10.0
        random_color = np.random.uniform(0, 0.9, size=(count, 3))
        random_color[0, 0] = np.random.uniform(0.8, 0.9)
        random_color[0, 1:] = 0.0
        random_color[1:, 0] = 0.0
        random_weights = np.random.uniform(0.01, 1, size=count - 1)
        random_weights = np.concatenate([[color], (1 - color) * random_weights / np.sum(random_weights)])
        new_img = gen_color_band(random_color, random_weights)
    if args.noisy:
        new_img += np.random.normal(loc=0, scale=0.03, size=new_img.shape)
        new_img = 1.0 - np.abs(1.0 - new_img)
        new_img = np.abs(new_img)
    images.append(new_img)
    colors.append(random_color)
    if (i+1) % 1000 == 0:
        print("Generating %d-th image, time used: %f" % (i+1, time.time() - start_time))

batch_img = np.stack(images, axis=0)
batch_color = np.stack(colors, axis=0)
if not os.path.isdir(args.dest):
    os.makedirs(args.dest)
np.savez(os.path.join(args.dest, 'batch%d.npz' % args.bn), images=batch_img, colors=batch_color, masks=masks)

