# __all__ = ['dataset_celeba', 'dataset_cifar', 'dataset_imagenet', 'dataset_mnist', 'dataset_svhn', 'dataset_mog', 'dataset_lsun', 'dataset_coco']

from .dataset_celeba import CelebADataset
from .dataset_cifar import CifarDataset
from .dataset_mnist import MnistDataset
from .dataset_svhn import SVHNDataset
from .dataset_lsun import LSUNDataset
from .dataset_color import DotsDataset