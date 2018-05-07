if __name__ == '__main__':
    from abstract_dataset import *
else:
    from .abstract_dataset import *
import numpy as np
import os, sys
import scipy.misc as misc

class CifarDataset(Dataset):
    def __init__(self, db_path='/data/cifar10', one_hot=False, use_edge=False):
        Dataset.__init__(self)
        self.data_dims = [32, 32, 3]
        self.range = [0.0, 1.0]
        self.batch_size = 100
        self.use_edge = use_edge
        self.name = "cifar"
        self.folder = "cifar"
        if self.use_edge:
            import cv2
        self.train_data = np.zeros((50000, 32, 32, 3))
        if one_hot:
            self.train_labels = np.zeros((50000, 10), dtype=np.float32)
        else:
            self.train_labels = np.zeros(50000, dtype=np.int8)

        for i in range(5):
            content = self.unpickle(os.path.join(db_path, 'data_batch_%d' % (i+1)))
            data, labels = content[b'data'], content[b'labels']
            self.train_data[i*10000:(i+1)*10000] = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))
            if one_hot:
                self.train_labels[np.arange(i*10000, (i+1)*10000, dtype=np.int32), labels] = 1.0
            else:
                self.train_labels[i*10000:(i+1)*10000] = labels

        if use_edge:
            edge_file = os.path.join(db_path, "canny_edge")
            if os.path.isfile(edge_file):
                self.edge_data = self.unpickle(edge_file)[b'edge']
            else:
                self.edge_data = self.compute_edges()
                self.pickle(edge_file, {'edge': self.edge_data})

        content = self.unpickle(os.path.join(db_path, 'test_batch'))
        data, labels = content[b'data'], content[b'labels']
        self.test_data = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))
        if one_hot:
            self.test_labels = np.zeros((10000, 10), dtype=np.float32)
            self.test_labels[np.arange(10000, dtype=np.int32), labels] = 1.0
        else:
            self.test_labels = np.array(labels, dtype=np.int8)

        # Load label names
        self.label_names = self.unpickle(os.path.join(db_path, 'batches.meta'))[b'label_names']

        self.train_batch_ptr = 0
        self.test_batch_ptr = 0
        self.train_data = np.clip(self.train_data / 255.0, a_min=0.0, a_max=1.0)
        self.test_data = np.clip(self.test_data / 255.0, a_min=0.0, a_max=1.0)
        self.train_size = self.train_data.shape[0]
        self.test_size = self.test_data.shape[0]

    @staticmethod
    def unpickle(filename):
        if sys.version_info[0] == 2:
            import cPickle
            fo = open(filename, 'rb')
            content = cPickle.load(fo)
            fo.close()
        else:
            import pickle
            with open(filename, 'rb') as fo:
                content = pickle.load(fo, encoding='bytes')
        return content

    def pickle(self, filename, save_dict):
        if sys.version_info[0] == 2:
            import cPickle
            cPickle.dump(save_dict, open(filename, 'wb'))
        else:
            import pickle
            pickle.dump(save_dict, open(filename, 'wb'))

    def compute_edges(self):
        edges = np.zeros((50000, 32, 32, 1), dtype=np.uint8)
        for i in range(self.train_data.shape[0]):
            img = (self.train_data[i] * 255).astype(np.uint8)
            edge = cv2.Canny(img, 100, 200) / 255
            edges[i] = np.reshape(edge, [32, 32, 1])
            if i % 100 == 0:
                print("Processing %d-th image" % i)
        return edges

    def move_train_batch_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.train_data.shape[0]:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
        return prev_batch_ptr

    def next_batch(self, batch_size=None):
        prev_batch_ptr = self.move_train_batch_ptr(batch_size)
        return self.train_data[prev_batch_ptr:self.train_batch_ptr]

    def next_labeled_batch(self, batch_size=None):
        prev_batch_ptr = self.move_train_batch_ptr(batch_size)
        return self.train_data[prev_batch_ptr:self.train_batch_ptr], self.train_labels[prev_batch_ptr:self.train_batch_ptr]

    def next_labeled_batch_with_edge(self, batch_size=None):
        prev_batch_ptr = self.move_train_batch_ptr(batch_size)
        return self.train_data[prev_batch_ptr:self.train_batch_ptr], \
               self.train_labels[prev_batch_ptr:self.train_batch_ptr], \
               self.edge_data[prev_batch_ptr:self.train_batch_ptr]

    def move_test_batch_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_batch_ptr = self.test_batch_ptr
        self.test_batch_ptr += batch_size
        if self.test_batch_ptr >self.test_data.shape[0]:
            self.test_batch_ptr = batch_size
            prev_batch_ptr = 0
        return prev_batch_ptr

    def next_labeled_test_batch(self, batch_size=None):
        prev_batch_ptr = self.move_test_batch_ptr(batch_size)
        return self.test_data[prev_batch_ptr:self.test_batch_ptr], self.test_labels[prev_batch_ptr:self.test_batch_ptr]

    def next_test_batch(self, batch_size=None):
        prev_batch_ptr = self.move_test_batch_ptr(batch_size)
        return self.test_data[prev_batch_ptr:self.test_batch_ptr]

    def batch_by_index(self, batch_start, batch_end):
        return self.train_data[batch_start:batch_end]

    def display(self, image):
        return np.clip(image, a_min=0.0, a_max=1.0)

    def label_name(self, index):
        return self.label_names[index]

    def reset(self):
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0

if __name__ == '__main__':
    # dataset = CifarDataset()
    # images, labels = dataset.next_labeled_batch(100)
    # print(images.shape, labels.shape)
    # for i in range(100):
    #     plt.subplot(10, 10, i+1)
    #     plt.title(dataset.label_names[labels[i]])
    #     plt.imshow(dataset.display(images[i]))
    # plt.show()
    from matplotlib import pyplot as plt
    dataset = CifarDataset(one_hot=True)
    images, labels, edges = dataset.next_labeled_batch_with_edge(100)
    print(images.shape, labels.shape, edges.shape)
    edges = edges[:, :-1, :-1, :] + edges[:, 1:, 1:, :]
    edges = np.minimum(edges, 1)
    for i in range(50):
        plt.subplot(10, 10, (2*i)+1)
        plt.title(dataset.label_names[np.argmax(labels[i])])
        plt.imshow(edges[i, :, :, 0], cmap='Greys')
        plt.subplot(10, 10, (2*i)+2)
        plt.imshow(images[i])
    plt.show()