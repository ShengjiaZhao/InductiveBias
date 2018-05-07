if __name__ == '__main__':
    from abstract_dataset import *
else:
    from .abstract_dataset import *
import time
import math

class DotsDataset(Dataset):
    def __init__(self, db_path='/data/dots2/size_8'):
        Dataset.__init__(self)
        self.data_dims = [64, 64, 3]
        self.name = "dots"
        self.batch_size = 100
        self.db_path = db_path

        self.db_files = os.listdir(self.db_path)
        self.train_db_files = self.db_files
        print(self.db_files)

        self.train_data_ptr = 0
        self.train_batch_ptr = -1
        self.train_batch, self.train_labels, self.masks = self.load_new_data()

        self.train_size = len(self.db_files) * 8192
        self.range = [0.0, 1.0]

    def load_new_data(self):
        self.train_batch_ptr += 1
        if self.train_batch_ptr == len(self.train_db_files):
            self.train_batch_ptr = 0
        filename = os.path.join(self.db_path, self.train_db_files[self.train_batch_ptr])
        result = np.load(filename)
        return result['images'], result['colors'], result['masks']

    def move_train_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_data_ptr = self.train_data_ptr
        self.train_data_ptr += batch_size
        if self.train_data_ptr > self.train_batch.shape[0]:
            self.train_data_ptr = batch_size
            prev_data_ptr = 0
            self.train_batch, self.train_labels, _ = self.load_new_data()
        return prev_data_ptr

    def next_batch(self, batch_size=None):
        return self.next_labeled_batch(batch_size)[0]

    def next_labeled_batch(self, batch_size=None):
        prev_data_ptr = self.move_train_ptr(batch_size)
        return self.train_batch[prev_data_ptr:self.train_data_ptr, :, :, :], \
               self.train_labels[prev_data_ptr:self.train_data_ptr]

    def reset(self):
        self.train_data_ptr = 0
        self.train_batch_ptr = -1
        self.train_batch, self.train_labels, _ = self.load_new_data()

    def eval_colors(self, arr):
        """ Only call this for the color dataset """
        colors = []
        for i in range(6):
            avg_color = np.average(self.masks[i] * arr, weights=np.tile(self.masks[i], [1, 1, 3]), axis=(0, 1))
            colors.append(avg_color)
        return np.stack(colors, axis=0)

    def plot_colors(self, ax, color_list):
        """ Only call this for the color dataset """
        angles = np.linspace(0, 2 * math.pi, 7)
        circle_locx = 0.5 + 0.36 * np.cos(angles)
        circle_locy = 0.5 + 0.36 * np.sin(angles)
        # fig = plt.figure(figsize=((64 + 2 * margin) / 10.0, (64 + 2 * margin) / 10.0), dpi=10)
        radius = 0.12
        for i in range(6):
            circle = plt.Circle((circle_locx[i], circle_locy[i]), radius, color=color_list[i])
            ax.add_artist(circle)

    @staticmethod
    def eval_size(arr):
        radius_list = []
        for i in range(arr.shape[0]):
            radius_list.append(DotsDataset.compute_radius(arr[i]))
        return np.array(radius_list)

    @staticmethod
    def compute_radius(img):
        min_cnt = 50
        size = -1
        for i in range(3):
            counts, _ = np.histogram(img[:, :, i], bins=50, range=(0, 1.0))
            if np.argmax(counts[:45]) < min_cnt:
                min_cnt = np.argmax(counts[:45])
                size = np.sum(counts[0:min_cnt + (50 - min_cnt) // 2])
            # print(counts, min_cnt, size)
        radius = math.sqrt(size / math.pi)
        return radius

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dataset = DotsDataset()
    images, labels = dataset.next_labeled_batch(100)
    for i in range(0, 16):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
    plt.show()

    plt.hist(dataset.eval_size(images), bins=20)
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
