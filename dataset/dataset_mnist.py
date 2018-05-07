from .abstract_dataset import *
from tensorflow.examples.tutorials.mnist import input_data


class MnistDataset(Dataset):
    def __init__(self, one_hot=True, binary=False):
        Dataset.__init__(self)
        data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MNIST_data')
        self.mnist = input_data.read_data_sets(data_file, one_hot=one_hot)
        self.name = "mnist"
        self.data_dims = [28, 28, 1]
        self.train_size = 50000
        self.test_size = 10000
        self.binary = binary
        self.range = [0.0, 1.0]

    def next_batch(self, batch_size):
        return self.next_labeled_batch(batch_size)[0]

    def next_labeled_batch(self, batch_size):
        image, label = self.mnist.train.next_batch(batch_size)
        image = np.reshape(image, (-1, 28, 28, 1))
        if self.binary:
            image = np.rint(image)
        return image, label

    def next_test_batch(self, batch_size):
        return self.next_labeled_test_batch(batch_size)[0]

    def next_labeled_test_batch(self, batch_size):
        image, label = self.mnist.test.next_batch(batch_size)
        image = np.reshape(image, (-1, 28, 28, 1))
        if self.binary:
            image = np.rint(image)
        return image, label

    def display(self, image):
        return np.clip(image, a_min=0.0, a_max=1.0)

    def reset(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


if __name__ == '__main__':
    binary_data = MnistDataset(binary=True)
    float_data = MnistDataset(binary=False)
    while True:
        binary_sample = binary_data.next_batch(100)
        float_sample = float_data.next_batch(100)
        for index in range(9):
            plt.subplot(3, 6, 2 * index + 1)
            plt.imshow(float_sample[index, :, :, 0].astype(np.float), cmap=plt.get_cmap('Greys'))
            plt.subplot(3, 6, 2 * index + 2)
            plt.imshow(binary_sample[index, :, :, 0].astype(np.float), cmap=plt.get_cmap('Greys'))
        plt.show()
