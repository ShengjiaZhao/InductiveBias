if __name__ == '__main__':
    from abstract_dataset import *
    from matplotlib import pyplot as plt
else:
    from .abstract_dataset import *

import tensorflow as tf
import pickle
import time


class CocoClassifyDataset(Dataset):
    def __init__(self, db_path="/ssd_data/coco/coco_label"):
        Dataset.__init__(self)
        self.batch_size = 100
        self.data_dims = [299, 299, 3]
        self.name = "coco"

        self.meta = pickle.load(open(os.path.join(db_path, "train2014/labels.p"), "rb"))

        self.train_files = os.listdir(os.path.join(db_path, "train2014"))
        self.test_files = os.listdir(os.path.join(db_path, "val2014"))

        self.train_files = [os.path.join(db_path, "train2014", file_name) for file_name in self.train_files
                            if file_name != "labels.p"]
        self.test_files = [os.path.join(db_path, "val2014", file_name) for file_name in self.test_files
                           if file_name != "labels.p"]
        print(self.train_files)
        print(self.test_files)

        self.train_data_ptr = 0
        self.train_batch_ptr = 0
        self.train_img, self.train_label = self.load_new_data()

        self.test_data_ptr = 0
        self.test_batch_ptr = 0
        self.test_img, self.test_label = self.load_new_test_data()

        self.num_classes = len(self.meta['names'])
        self.color_map = np.random.rand(256, 3)

    def load_new_data(self):
        if len(self.train_files) == 0:
            return None, None
        data = np.load(self.train_files[self.train_batch_ptr])
        self.train_batch_ptr += 1
        if self.train_batch_ptr == len(self.train_files):
            self.train_batch_ptr = 0
        return data['images'], data['labels']

    def load_new_test_data(self):
        if len(self.test_files) == 0:
            return None, None
        data = np.load(self.test_files[self.test_batch_ptr])
        self.test_batch_ptr += 1
        if self.test_batch_ptr == len(self.test_files):
            self.test_batch_ptr = 0
        return data['images'], data['labels']

    def next_labeled_batch(self, batch_size=None):
        prev_ptr = self.move_train_ptr(batch_size)
        return self.train_img[prev_ptr:self.train_data_ptr], \
               self.train_label[prev_ptr:self.train_data_ptr]

    def next_labeled_test_batch(self, batch_size=None):
        prev_ptr = self.move_test_ptr(batch_size)
        return self.test_img[prev_ptr:self.test_data_ptr], \
               self.test_label[prev_ptr:self.test_data_ptr]

    def move_train_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.train_data_ptr
        self.train_data_ptr += batch_size
        if self.train_data_ptr > self.train_img.shape[0]:
            self.train_data_ptr = batch_size
            prev_ptr = 0
            self.train_img, self.train_label = self.load_new_data()
        return prev_ptr

    def move_test_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.test_data_ptr
        self.test_data_ptr += batch_size
        if self.test_data_ptr > self.test_img.shape[0]:
            self.test_data_ptr = batch_size
            prev_ptr = 0
            self.test_img, self.test_label = self.load_new_test_data()
        return prev_ptr

    def label_name(self, index):
        return self.meta['names'][index]

    def display(self, image):
        print(np.max(image), np.min(image))
        return np.clip(image, 0.0, 1.0)

    def display_tf(self, image):
        rescaled = tf.divide(image + 1.0, 2.0)
        return tf.clip_by_value(rescaled, 0.0, 1.0)


if __name__ == '__main__':
    start_time = time.time()
    dataset = CocoClassifyDataset()
    print("%d - %s" % (len(dataset.meta['names']), str(dataset.meta['names'])))
    print("Finished, loading used %fs" % (time.time() - start_time))
    for j in range(100):
        start_time = time.time()
        images, labels = dataset.next_labeled_batch(100)
        print("Time used %f" % (time.time() - start_time))
        for i in range(0, 16):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i])
            names = [dataset.label_name(index) for index in range(labels.shape[1]) if labels[i, index] == 1]
            plt.title(" ".join(names))
        plt.show()

        for i in range(49):
            images, labels = dataset.next_labeled_batch(100)