if __name__ == '__main__':
    from abstract_dataset import *
    from color_data.utils import gen_combi
else:
    from .abstract_dataset import *
    from .color_data.utils import gen_combi

import time
import math
import cv2
from matplotlib import colors as pltcolor


class DotsDataset2(Dataset):
    def __init__(self, params=('46000',)):
        Dataset.__init__(self)
        self.data_dims = [64, 64, 3]
        self.name = "dots"
        self.batch_size = 100
        self.params = params

        self.train_ptr = 0
        self.train_cache = []
        self.max_size = 200000

        self.range = [0.0, 1.0]

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.train_ptr
        self.train_ptr += batch_size
        if self.train_ptr > self.max_size:
            prev_ptr = 0
            self.train_ptr = batch_size
        while self.train_ptr > len(self.train_cache):
            self.train_cache.append(gen_combi(np.random.choice(self.params)))
        return np.stack(self.train_cache[prev_ptr:self.train_ptr], axis=0)

    def reset(self):
        self.train_ptr = 0

    @staticmethod
    def eval_color_gap(arr):
        count_list = []
        for i in range(arr.shape[0]):
            count_list.append(DotsDataset2.compute_color_gap(arr[i]))
        return np.array(count_list)

    @staticmethod
    def eval_count(arr):
        count_list = []
        for i in range(arr.shape[0]):
            count_list.append(DotsDataset2.compute_count(arr[i]))
        return np.array(count_list)

    @staticmethod
    def eval_size(arr):
        radius_list = []
        for i in range(arr.shape[0]):
            radius_list.append(DotsDataset2.compute_radius(arr[i]))
        return np.array(radius_list)

    @staticmethod
    def eval_color_proportion(arr):
        prop_list = []
        for i in range(arr.shape[0]):
            prop_list.append(DotsDataset2.compute_proportion(arr[i]))
        return np.array(prop_list)

    # @staticmethod
    # def compute_radius(img):
    #     min_cnt = 50
    #     size = -1
    #     for i in range(3):
    #         counts, _ = np.histogram(img[:, :, i], bins=50, range=(0, 1.0))
    #         if np.argmax(counts[:45]) < min_cnt:
    #             min_cnt = np.argmax(counts[:45])
    #             size = np.sum(counts[0:min_cnt + (50 - min_cnt) // 2])
    #         # print(counts, min_cnt, size)
    #     radius = math.sqrt(size / math.pi)
    #     return radius

    @staticmethod
    def compute_radius(img):
        img = np.reshape(img, [-1, 3])
        size = float(len(np.argwhere(np.sum(img, axis=1) < 2.7)))
        radius = math.sqrt(size / math.pi)
        return radius / 32.0

    @staticmethod
    def eval_location(arr):
        location_list = []
        for i in range(arr.shape[0]):
            location_list.append(DotsDataset2.compute_location(arr[i]))
        return np.stack(location_list, axis=0)

    @staticmethod
    def compute_proportion(img):
        img = np.reshape(img, [-1, 3])
        colors = img[np.argwhere(np.sum(img, axis=1) < 2.7)[:, 0]]
        reds = np.argwhere(colors[:, 0] > np.max(colors[:, 1:], axis=1))
        return float(reds.shape[0]) / colors.shape[0]

    @staticmethod
    def compute_count(arr):
        cimg = 255 - (arr*255).astype(np.uint8)
        img = cv2.cvtColor(cimg, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0, 0)
        img = cv2.medianBlur(img, 5)
        # img = cv2.GaussianBlur(img,(3,3),0,0)
        # _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        # plt.show()
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10,
                                   param1=40, param2=3, minRadius=4, maxRadius=7)
        if circles is None:
            return 0
        else:
            return len(circles[0])

    @staticmethod
    def compute_location(img):
        colors = np.argwhere(np.sum(img, axis=2) < 2.7)
        location = np.mean(colors, axis=0) / 32.0 - 1.0
        return location

    @staticmethod
    def compute_color_gap(img):
        img = np.reshape(img, [-1, 3])
        colors = img[np.argwhere(np.sum(img, axis=1) < 2.7)[:, 0]]
        colors = np.mean(colors, axis=0, keepdims=True)
        return pltcolor.rgb_to_hsv(colors)[0, 0]

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dataset = DotsDataset2()
    images = dataset.next_batch(100)
    for i in range(0, 16):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
    plt.show()
    plt.subplot(2, 2, 1)
    plt.hist(dataset.eval_location(images)[:, 0], range=(-0.5, 0.5), bins=30)
    plt.subplot(2, 2, 2)
    plt.hist(dataset.eval_location(images)[:, 1], range=(-0.5, 0.5), bins=30)
    plt.subplot(2, 2, 3)
    plt.hist(dataset.eval_color_proportion(images), range=(0, 1), bins=30)
    plt.subplot(2, 2, 4)
    plt.hist(dataset.eval_size(images), range=(0, 1), bins=30)
    plt.show()
    # for i in range(0, 16):
    #     plt.subplot(4, 4, i+1)
    #     dataset.plot_colors(plt.gca(), labels[i])
    # plt.show()
    #
    # for i in range(0, 16):
    #     plt.subplot(4, 4, i+1)
    #     print(dataset.eval_colors(images[i]))
    #     dataset.plot_colors(plt.gca(), dataset.eval_colors(images[i]))
    # plt.show()
