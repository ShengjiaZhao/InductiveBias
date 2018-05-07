if __name__ == '__main__':
    from abstract_dataset import *
else:
    from .abstract_dataset import *
import time

class ImagenetDataset(Dataset):
    def __init__(self, use_edge=False, db_path='/data/data/imagenet/train64'):
        Dataset.__init__(self)
        self.data_dims = [64, 64, 3]
        self.name = "imagenet"
        self.batch_size = 100
        self.db_path = db_path
        self.use_edge = use_edge
        if self.use_edge:
            import cv2
        synset_file = open(os.path.join(self.db_path, "meta.txt"), "r")
        self.synset = []
        while True:
            line = synset_file.readline().split()
            if len(line) < 3:
                break
            self.synset.append([line[1], ' '.join(line[2:])])
        self.num_classes = len(self.synset)
        print("Total classes %d" % self.num_classes)

        self.db_files = os.listdir(self.db_path)
        self.db_files.remove("meta.txt")
        self.train_db_files = self.db_files[5:]
        self.edge_cache = []
        self.test_db_files = self.db_files[0:5]

        self.train_data_ptr = 0
        self.train_batch_ptr = -1
        self.train_batch, self.train_labels = self.load_new_data()
        if use_edge:
            self.edge_batch = self.compute_edges()

        self.test_data_ptr = 0
        self.test_batch_ptr = -1
        self.test_batch, self.test_labels = self.load_new_test_data()

        self.train_size = len(self.train_db_files) * 10000
        self.test_size = len(self.test_db_files) * 10000
        self.range = [-1.0, 1.0]

    def compute_edges(self):
        if len(self.edge_cache) > self.train_batch_ptr:
            return self.edge_cache[self.train_batch_ptr]
        edges = np.zeros((10000, 64, 64, 1), dtype=np.uint8)
        for i in range(self.train_batch.shape[0]):
            img = ((self.train_batch[i] + 1.0) / 2.0 * 255).astype(np.uint8)
            edge = cv2.Canny(img, 80, 160) / 255
            edges[i] = np.reshape(edge, [64, 64, 1])
        self.edge_cache.append(edges)
        return edges

    def load_new_data(self):
        self.train_batch_ptr += 1
        if self.train_batch_ptr == len(self.train_db_files):
            self.train_batch_ptr = 0
        filename = os.path.join(self.db_path, self.train_db_files[self.train_batch_ptr])
        result = np.load(filename)
        return result['images'], result['labels']

    def load_new_test_data(self):
        self.test_batch_ptr += 1
        if self.test_batch_ptr == len(self.test_db_files):
            self.test_batch_ptr = 0
        filename = os.path.join(self.db_path, self.test_db_files[self.test_batch_ptr])
        result = np.load(filename)
        return result['images'], result['labels']

    def move_train_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_data_ptr = self.train_data_ptr
        self.train_data_ptr += batch_size
        if self.train_data_ptr > self.train_batch.shape[0]:
            self.train_data_ptr = batch_size
            prev_data_ptr = 0
            self.train_batch, self.train_labels = self.load_new_data()
            if self.use_edge:
                self.edge_batch = self.compute_edges()
        return prev_data_ptr

    def move_test_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_data_ptr = self.test_data_ptr
        self.test_data_ptr += batch_size
        if self.test_data_ptr > self.test_batch.shape[0]:
            self.test_data_ptr = batch_size
            prev_data_ptr = 0
            self.test_batch, self.test_labels = self.load_new_test_data()
        return prev_data_ptr

    def next_batch(self, batch_size=None):
        return self.next_labeled_batch(batch_size)[0]

    def next_labeled_batch(self, batch_size=None):
        prev_data_ptr = self.move_train_ptr(batch_size)
        return self.train_batch[prev_data_ptr:self.train_data_ptr, :, :, :], \
               self.train_labels[prev_data_ptr:self.train_data_ptr]

    def next_labeled_batch_with_edge(self, batch_size=None):
        prev_data_ptr = self.move_train_ptr(batch_size)
        return self.train_batch[prev_data_ptr:self.train_data_ptr, :, :, :], \
               self.train_labels[prev_data_ptr:self.train_data_ptr], \
               self.edge_batch[prev_data_ptr:self.train_data_ptr]

    def next_labeled_test_batch(self, batch_size=None):
        prev_data_ptr = self.move_test_ptr(batch_size)
        return self.test_batch[prev_data_ptr:self.test_data_ptr, :, :, :], \
               self.test_labels[prev_data_ptr:self.test_data_ptr]

    def next_test_batch(self, batch_size=None):
        return self.next_labeled_test_batch(batch_size)[0]

    def display(self, image):
        rescaled = np.divide(image + 1.0, 2.0)
        return np.clip(rescaled, a_min=0.0, a_max=1.0)

    def display_tf(self, image):
        rescaled = tf.divide(image + 1.0, 2.0)
        return tf.clip_by_value(rescaled, 0.0, 1.0)

    def reset(self):
        self.batch_ptr = 0

    def label_name(self, index):
        return self.synset[index][1]

if __name__ == '__main__':
    dataset = ImagenetDataset(use_edge=True)
    for j in range(100):
        for k in range(200):
            start_time = time.time()
            images, labels, edges = dataset.next_labeled_batch_with_edge(1000)
            if k % 10 == 0:
                print("Batch %d" % k)
                print("Time used %f" % (time.time() - start_time))
        for i in range(0, 8):
            plt.subplot(4, 4, (2 * i) + 1)
            plt.imshow(dataset.display(images[i]))
            plt.gca().set_title(dataset.label_name(labels[i]))
            plt.subplot(4, 4, (2 * i) + 2)
            plt.imshow(edges[i, :, :, 0], cmap='Greys')
        plt.show()