if __name__ == '__main__':
    from abstract_dataset import *
    from matplotlib import pyplot as plt
else:
    from .abstract_dataset import *

import tensorflow as tf
import pickle
import time

class CocoDataset(Dataset):
    def __init__(self, use_edge=False, db_path="/data/data/coco"):
        Dataset.__init__(self)
        self.batch_size = 100
        self.data_dims = [64, 64, 3]
        self.name = "coco"
        self.use_edge = use_edge
        if self.use_edge:
            import cv2
        self.categories = pickle.load(open(os.path.join(db_path, "train2014_64/labels.p"), "rb"))['names']

        self.train_files = os.listdir(os.path.join(db_path, "train2014_64"))
        self.test_files = os.listdir(os.path.join(db_path, "val2014_64"))
        self.train_files = [os.path.join(db_path, "train2014_64", file_name) for file_name in self.train_files if file_name != "labels.p"]
        self.test_files = [os.path.join(db_path, "val2014_64", file_name) for file_name in self.test_files if file_name != "labels.p"]
        self.edge_cache = []
        print(self.train_files)
        print(self.test_files)

        self.train_data_ptr = 0
        self.train_batch_ptr = 0
        self.train_img, self.train_label, self.train_mask = self.load_new_data()
        if use_edge:
            self.train_edge = self.compute_edges()

        self.test_data_ptr = 0
        self.test_batch_ptr = 0
        self.test_img, self.test_label, self.test_mask = self.load_new_test_data()

        self.num_classes = len(self.categories)
        self.color_map = np.random.rand(256, 3)

    def compute_edges(self):
        if len(self.edge_cache) > self.train_batch_ptr:
            return self.edge_cache[self.train_batch_ptr]
        edges = np.zeros((10000, 64, 64, 1), dtype=np.uint8)
        for i in range(self.train_img.shape[0]):
            img = ((self.train_img[i] + 1.0) / 2.0 * 255).astype(np.uint8)
            edge = cv2.Canny(img, 80, 160) / 255
            edges[i] = np.reshape(edge, [64, 64, 1])
        self.edge_cache.append(edges)
        return edges

    def load_new_data(self):
        data = np.load(self.train_files[self.train_batch_ptr])
        self.train_batch_ptr += 1
        if self.train_batch_ptr == len(self.train_files):
            self.train_batch_ptr = 0
        return data['images'], data['labels'], np.expand_dims(data['masks'], len(data['masks'])-1)

    def load_new_test_data(self):
        data = np.load(self.test_files[self.test_batch_ptr])
        self.test_batch_ptr += 1
        if self.test_batch_ptr == len(self.test_files):
            self.test_batch_ptr = 0
        return data['images'], data['labels'], np.expand_dims(data['masks'], len(data['masks'])-1)

    def next_batch(self, batch_size=None):
        return self.next_batch_with_mask(batch_size)[0]

    def next_labeled_batch(self, batch_size=None):
        return self.next_batch_with_mask(batch_size)[0:2]

    def next_labeled_batch_with_edge(self, batch_size=None):
        prev_ptr = self.move_train_ptr(batch_size)
        return self.train_img[prev_ptr:self.train_data_ptr], \
               self.train_label[prev_ptr:self.train_data_ptr], \
               self.train_edge[prev_ptr:self.train_data_ptr],

    def next_labeled_batch_with_all(self, batch_size=None):
        prev_ptr = self.move_train_ptr(batch_size)
        return self.train_img[prev_ptr:self.train_data_ptr], \
               self.train_label[prev_ptr:self.train_data_ptr], \
               self.train_edge[prev_ptr:self.train_data_ptr], \
               self.train_mask[prev_ptr:self.train_data_ptr]

    def next_batch_with_mask(self, batch_size=None):
        prev_ptr = self.move_train_ptr(batch_size)
        return self.train_img[prev_ptr:self.train_data_ptr, :, :, :], \
               self.train_label[prev_ptr:self.train_data_ptr, :], \
               self.train_mask[prev_ptr:self.train_data_ptr, :, :]

    def next_test_batch(self, batch_size=None):
        return self.next_test_batch_with_mask(batch_size)[0]

    def next_labeled_test_batch(self, batch_size=None):
        return self.next_test_batch_with_mask(batch_size)[0:2]

    def next_test_batch_with_mask(self, batch_size=None):
        prev_ptr = self.move_test_ptr(batch_size)
        return self.test_img[prev_ptr:self.test_data_ptr], \
               self.test_label[prev_ptr:self.test_data_ptr], \
               self.test_mask[prev_ptr:self.test_data_ptr]

    def move_train_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.train_data_ptr
        self.train_data_ptr += batch_size
        if self.train_data_ptr > self.train_img.shape[0]:
            self.train_data_ptr = batch_size
            prev_ptr = 0
            self.train_img, self.train_label, self.train_mask = self.load_new_data()
            if self.use_edge:
                self.train_edge = self.compute_edges()
        return prev_ptr

    def move_test_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.test_data_ptr
        self.test_data_ptr += batch_size
        if self.test_data_ptr > self.test_img.shape[0]:
            self.test_data_ptr = batch_size
            prev_ptr = 0
            self.test_img, self.test_label, self.test_mask = self.load_new_test_data()
        return prev_ptr

    def label_name(self, index):
        return self.categories[index]

    def display(self, image, mask=None):
        return self.display_image(image, mask)

    def display_tf(self, image):
        rescaled = tf.divide(image + 1.0, 2.0)
        return tf.clip_by_value(rescaled, 0.0, 1.0)

    def display_mask_tf(self, mask):
        color_map_tf = tf.constant(self.color_map, dtype=tf.float32)
        colored = tf.gather_nd(color_map_tf, tf.cast(mask, tf.int32))
        return colored

    """ Transform image to displayable """
    def display_image(self, image, mask=None):
        if mask is None:
            return np.clip((image + 1.0) / 2.0, 0.0, 1.0)
        else:
            return np.clip((image + 1.0) / 2.0, 0.0, 1.0) * 0.8 + self.display_mask(mask) * 0.2

    def display_mask(self, mask):
        # canvas = np.zeros((mask.shape[0] * 2, mask.shape[1] * 2, 3), dtype=np.float32)
        colored = self.color_map[mask]
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(colored)
        # ax.text(x=0.5, y=0.6, s="Hello World")
        # fig.canvas.draw()
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return colored

if __name__ == '__main__':
    # dataset = CocoDataset(use_edge=True)
    # while True:
    #     batch, label, mask = dataset.next_batch_with_mask(100)
    #     for i in range(25):
    #         plt.subplot(5, 10, (2 * i) + 1)
    #         plt.imshow(dataset.display(batch[i]))
    #         plt.title(" ".join([dataset.label_name(index) for index in np.argwhere(label[i] == 1)[:, 0]]))
    #         plt.subplot(5, 10, (2 * i) + 2)
    #         plt.imshow(dataset.display_mask(mask[i]))
    #     plt.show()
    #     for i in range(99):
    #         dataset.next_batch(100)

    dataset = CocoDataset(use_edge=True)
    for j in range(100):
        for k in range(200):
            start_time = time.time()
            images, labels, edges = dataset.next_labeled_batch_with_edge(1000)
            print("Batch %d" % k)
            print("Time used %f" % (time.time() - start_time))
            for i in range(0, 8):
                plt.subplot(4, 4, (2 * i) + 1)
                plt.imshow(dataset.display(images[i]))
                plt.title(" ".join([dataset.label_name(index) for index in np.argwhere(labels[i] == 1)[:, 0]]))
                plt.subplot(4, 4, (2 * i) + 2)
                plt.imshow(edges[i, :, :, 0], cmap='Greys')
            plt.show()