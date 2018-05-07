import numpy as np
import scipy.misc as misc
from matplotlib import pyplot as plt

reader = np.load('lsun_fake.npz')
bedrooms = reader['bedroom']
churches = reader['church']
conferences = reader['conference']
dinings = reader['dining']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(bedrooms[i])
    plt.axis('off')
plt.show()

reader = np.load('lsun_real.npz')
bedrooms = reader['bedroom']
churches = reader['church']
conferences = reader['conference']
dinings = reader['dining']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(bedrooms[i])
    plt.axis('off')
plt.show()