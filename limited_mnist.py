from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# Train on limited data
class LimitedMnist:
    def __init__(self, size):
        self.data_ptr = 0
        self.size = size
        self.full_mnist = input_data.read_data_sets('mnist_data')
        assert size <= self.full_mnist.train.images.shape[0]
        self.data = self.full_mnist.train.images
        np.random.shuffle(self.data)
        self.data = self.data[:size]

    def next_batch(self, batch_size):
        assert batch_size <= self.size
        prev_ptr = self.data_ptr
        self.data_ptr += batch_size
        if self.data_ptr > self.size:
            self.data_ptr -= self.size
            return np.concatenate([self.data[prev_ptr:], self.data[:self.data_ptr]], axis=0)
        else:
            return self.data[prev_ptr:self.data_ptr]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    limited_mnist = LimitedMnist(200)
    while True:
        image_x = limited_mnist.next_batch(150).reshape([-1, 28, 28])
        for i in range(100):
            plt.subplot(10, 10, i+1)
            plt.imshow(image_x[i])
        plt.show()