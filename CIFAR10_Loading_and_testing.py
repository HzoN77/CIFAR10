#
#
# Initial tests on CIFAR10 data. how to read/load data and handle it.

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


cifar_path = './cifar-10-batches-py/'


import numpy as np
import matplotlib.pyplot as plt
first_file = unpickle(cifar_path + 'data_batch_1')
plt.imshow(np.swapaxes(np.reshape(first_file[b'data'][40], (32,32,3), order='F'), 0, 1))
plt.show()

# new update
