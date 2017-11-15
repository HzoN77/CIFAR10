import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

def view_test_data(X_data, y_data, model):
    ncols = 10
    nrows = 5
    idx = random.sample(range(1, 9999), ncols*nrows)
    X = X_data[idx]
    y = y_data[idx]
    y_pred = model.predict(X)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    ctr = 0
    for ax in axes.flatten():
        im = ax.imshow(X[ctr])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title('Pred:' + str(np.argmax(y_pred[ctr])) + ' True:' + str(np.argmax(y[ctr])))
        ctr += 1
    return fig


def load_in_CIFAR(file, full_dict=False):
    CIFAR_input_size = (32, 32, 3)
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    # Transform the images to correct RGB-representation.
    y_label = dict[b'labels']
    X_flat = dict[b'data']
    X = [np.swapaxes(np.reshape(x, CIFAR_input_size, order='F'), 0, 1) for x in X_flat]
    y = y_label

    # Optional outputs the full dict. Additional info in the dict.
    if full_dict == True:
        return X, y, dict

    return X, y