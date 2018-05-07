if __name__ == '__main__':
    from abstract_dataset import *
    from matplotlib import pyplot as plt
else:
    from .abstract_dataset import *

import tensorflow as tf
import pickle
import time


class CocoTransferDataset(Dataset):
    def __init__(self, db_path="/ssd_data/coco/coco_bbox_transfer15_65_299"):
        Dataset.__init__(self)
        self.batch_size = 100
        self.data_dims = [299, 299, 3]
        self.name = "coco"

        self.meta = pickle.load(open(os.path.join(db_path, "labels.p"), "rb"))

        self.train_files = os.listdir(os.path.join(db_path, "train2014"))
        self.test_files = os.listdir(os.path.join(db_path, "val2014"))
        self.train_files = [os.path.join(db_path, "train2014", file_name) for file_name in self.train_files
                            if file_name != "labels.p"]
        self.test_files = [os.path.join(db_path, "val2014", file_name) for file_name in self.test_files
                           if file_name != "labels.p"]
        self.train_files.sort()
        self.test_files.sort()
        print(self.train_files)
        print(self.test_files)

        self.train_data_ptr = 0
        self.train_batch_ptr = 0
        self.train_img, self.train_img_mask, self.train_label1, self.train_label2 = self.load_new_data()

        self.test_data_ptr = 0
        self.test_batch_ptr = 0
        self.test_img, self.test_img_mask, self.test_label1, self.test_label2 = self.load_new_test_data()

        self.num_classes = [len(self.meta['train_names']), len(self.meta['transfer_names'])]
        self.color_map = np.random.rand(256, 3)

    def load_new_data(self):
        if len(self.train_files) == 0:
            return None, None, None
        data = np.load(self.train_files[self.train_batch_ptr])
        self.train_batch_ptr += 1
        if self.train_batch_ptr == len(self.train_files):
            self.train_batch_ptr = 0
        return data['images'], data['masked_images'], data['labels1'], data['labels2']

    def load_new_test_data(self):
        if len(self.test_files) == 0:
            return None, None, None
        data = np.load(self.test_files[self.test_batch_ptr])
        self.test_batch_ptr += 1
        if self.test_batch_ptr == len(self.test_files):
            self.test_batch_ptr = 0
        return data['images'], data['masked_images'], data['labels1'], data['labels2']

    def next_batch(self, batch_size=None):
        return self.next_transfer_batch(batch_size)[0]

    def next_transfer_batch(self, batch_size=None):
        prev_ptr = self.move_train_ptr(batch_size)
        return self.train_img[prev_ptr:self.train_data_ptr], \
               self.train_img_mask[prev_ptr:self.train_data_ptr], \
               self.train_label1[prev_ptr:self.train_data_ptr], \
               self.train_label2[prev_ptr:self.train_data_ptr],

    def next_test_batch(self, batch_size=None):
        return self.next_transfer_test_batch(batch_size)[0]

    def next_transfer_test_batch(self, batch_size=None):
        prev_ptr = self.move_test_ptr(batch_size)
        return self.test_img[prev_ptr:self.test_data_ptr], \
               self.test_img_mask[prev_ptr:self.test_data_ptr], \
               self.test_label1[prev_ptr:self.test_data_ptr], \
               self.test_label2[prev_ptr:self.test_data_ptr]

    def move_train_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.train_data_ptr
        self.train_data_ptr += batch_size
        if self.train_data_ptr > self.train_img.shape[0]:
            self.train_data_ptr = batch_size
            prev_ptr = 0
            self.train_img, self.train_img_mask, self.train_label1, self.train_label2 = self.load_new_data()
        return prev_ptr

    def move_test_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.test_data_ptr
        self.test_data_ptr += batch_size
        if self.test_data_ptr > self.test_img.shape[0]:
            self.test_data_ptr = batch_size
            prev_ptr = 0
            self.test_img, self.test_img_mask, self.test_label1, self.test_label2 = self.load_new_test_data()
        return prev_ptr

    def label_name(self, index, type='all'):
        if type == 'all':
            return self.meta['names'][index]
        elif type == 'train':
            return self.meta['train_names'][index]
        elif type == 'transfer':
            return self.meta['transfer_names'][index]
        else:
            return 'unknown type'

    def reset_ptr(self):
        self.train_data_ptr = 0
        self.train_batch_ptr = 0
        self.train_img, self.train_label1, self.train_label2 = self.load_new_data()

        self.test_data_ptr = 0
        self.test_batch_ptr = 0
        self.test_img, self.test_label1, self.test_label2 = self.load_new_test_data()

    def display(self, image):
        print(np.max(image), np.min(image))
        return np.clip(image, 0.0, 1.0)

    def display_tf(self, image):
        rescaled = tf.divide(image + 1.0, 2.0)
        return tf.clip_by_value(rescaled, 0.0, 1.0)


if __name__ == '__main__':
    start_time = time.time()
    dataset = CocoTransferDataset()
    print("Finished, loading used %fs" % (time.time() - start_time))
    for j in range(100):
        start_time = time.time()
        images, images_mask, labels1, labels2 = dataset.next_transfer_batch(100)
        print("Time used %f" % (time.time() - start_time))
        for i in range(0, 16):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i])
            names1 = [dataset.label_name(index, type='train') for index in range(labels1.shape[1]) if labels1[i, index] == 1]
            names2 = [dataset.label_name(index, type='transfer') for index in range(labels2.shape[1]) if labels2[i, index] == 1]
            plt.title(" ".join(names1) + '/' + " ".join(names2))
        plt.show()