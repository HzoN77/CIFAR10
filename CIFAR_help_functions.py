import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

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


def create_AlexNet(network_input_shape=(224, 224, 3), num_classes=None):
    alexnet = Sequential()
    # Conv 1

    alexnet.add(Conv2D(input_shape=network_input_shape, filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu'))
    alexnet.add(BatchNormalization())
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv 2
    alexnet.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
    alexnet.add(BatchNormalization())
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # COnv 3
    alexnet.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

    # Conv 4
    alexnet.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

    # Conv 5'th layer.
    alexnet.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # 6th layer: Flatten and fully connected
    alexnet.add(Flatten())
    alexnet.add(Dense(4096, activation='relu'))
    alexnet.add(Dropout(0.5))

    # 7th layer- Dense
    alexnet.add(Dense(4096, activation='relu'))
    alexnet.add(Dropout(0.5))

    # 8th layer: Output layer
    alexnet.add(Dense(num_classes, activation='softmax'))

    alexnet.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

    return alexnet

def create_fake_VGG16(CIFAR_input_size, num_classes):
    model = Sequential([
        # Block 1: In 32x32, out 16x16.
        Conv2D(64, (3, 3), input_shape=CIFAR_input_size, padding='same', activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Dropout(0.5),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Block 2: In 16x16, out 8x8.
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Dropout(0.5),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Block 3: in 8x8, out 4x4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Dropout(0.5),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Flatten, and Dense layers for output.
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
