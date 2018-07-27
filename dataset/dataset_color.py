if __name__ == '__main__':
    from abstract_dataset import *
else:
    from .abstract_dataset import *
import time
import math
import cv2
from matplotlib import colors as pltcolor

class DotsDataset(Dataset):
    def __init__(self, db_path=('/data/dots/combir_100',)):
        Dataset.__init__(self)
        self.data_dims = [64, 64, 3]
        self.name = "dots"
        self.batch_size = 100
        self.db_path = db_path

        self.db_files = [os.listdir(path) for path in db_path]
        assert np.min([len(files) for files in self.db_files]) == np.max([len(files) for files in self.db_files])
        self.train_db_files = self.db_files
        print(self.db_files)

        self.train_data_ptr = 0
        self.train_batch_ptr = -1
        self.batch_cache = {}
        self.train_batch = self.load_new_data()

        self.train_size = len(self.db_files) * 8192
        self.range = [0.0, 1.0]

    def load_new_data(self):
        self.train_batch_ptr += 1
        if self.train_batch_ptr == len(self.train_db_files[0]):
            self.train_batch_ptr = 0

        if self.train_batch_ptr not in self.batch_cache:
            images_list = []
            for dtype in range(len(self.train_db_files)):
                filename = os.path.join(self.db_path[dtype], self.train_db_files[dtype][self.train_batch_ptr])
                result = np.load(filename)
                images_list.append(result['images'])
            images = np.concatenate(images_list, axis=0)
            # colors = np.concatenate(colors_list, axis=0)
            perm = np.random.permutation(range(images.shape[0]))
            images = images[perm]
            # colors = colors[perm]
            self.batch_cache[self.train_batch_ptr] = {'images': images}

        return self.batch_cache[self.train_batch_ptr]['images']

    def move_train_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_data_ptr = self.train_data_ptr
        self.train_data_ptr += batch_size
        if self.train_data_ptr > self.train_batch.shape[0]:
            self.train_data_ptr = batch_size
            prev_data_ptr = 0
            self.train_batch = self.load_new_data()
        return prev_data_ptr

    def next_batch(self, batch_size=None):
        prev_data_ptr = self.move_train_ptr(batch_size)
        return self.train_batch[prev_data_ptr:self.train_data_ptr, :, :, :]

    # def next_labeled_batch(self, batch_size=None):
    #     prev_data_ptr = self.move_train_ptr(batch_size)
    #     return self.train_batch[prev_data_ptr:self.train_data_ptr, :, :, :], \
    #            self.train_labels[prev_data_ptr:self.train_data_ptr]

    def reset(self):
        self.train_data_ptr = 0
        self.train_batch_ptr = -1
        self.train_batch = self.load_new_data()

    # def eval_colors(self, arr):
    #     """ Only call this for the color dataset """
    #     colors = []
    #     for i in range(6):
    #         avg_color = np.average(self.masks[i] * arr, weights=np.tile(self.masks[i], [1, 1, 3]), axis=(0, 1))
    #         colors.append(avg_color)
    #     return np.stack(colors, axis=0)

    # def plot_colors(self, ax, color_list):
    #     """ Only call this for the color dataset """
    #     angles = np.linspace(0, 2 * math.pi, 7)
    #     circle_locx = 0.5 + 0.36 * np.cos(angles)
    #     circle_locy = 0.5 + 0.36 * np.sin(angles)
    #     # fig = plt.figure(figsize=((64 + 2 * margin) / 10.0, (64 + 2 * margin) / 10.0), dpi=10)
    #     radius = 0.12
    #     for i in range(6):
    #         circle = plt.Circle((circle_locx[i], circle_locy[i]), radius, color=color_list[i])
    #         ax.add_artist(circle)

    @staticmethod
    def eval_color_gap(arr):
        count_list = []
        for i in range(arr.shape[0]):
            count_list.append(DotsDataset.compute_color_gap(arr[i]))
        return np.array(count_list)

    @staticmethod
    def eval_count(arr):
        count_list = []
        for i in range(arr.shape[0]):
            count_list.append(DotsDataset.compute_count(arr[i]))
        return np.array(count_list)

    @staticmethod
    def eval_size(arr):
        radius_list = []
        for i in range(arr.shape[0]):
            radius_list.append(DotsDataset.compute_radius(arr[i]))
        return np.array(radius_list)

    @staticmethod
    def eval_color_proportion(arr):
        prop_list = []
        for i in range(arr.shape[0]):
            prop_list.append(DotsDataset.compute_proportion(arr[i]))
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
            location_list.append(DotsDataset.compute_location(arr[i]))
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
    dataset = DotsDataset()
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
