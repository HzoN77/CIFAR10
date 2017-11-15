#
#
# Initial tests on CIFAR10 data. how to read/load data and handle it.

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt



def load_in_CIFAR(file, full_dict=False):
    import pickle
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


def view_test_data(X_data, y_data):
    import random
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
        ax.set_title('Pred:' + str(np.argmax(y_pred[ctr])) + ' True:' + str(np.argmax(y_data[idx])))
        ctr += 1
    return fig


# Hyper params
batch_size = 128
num_classes = 10
epochs = 30
CIFAR_input_size = (32, 32, 3)


# Read in the CIFAR data
cifar_path = './cifar-10-batches-py/'
X1, y1 = load_in_CIFAR(cifar_path + 'data_batch_1')
X2, y2 = load_in_CIFAR(cifar_path + 'data_batch_2')
X3, y3 = load_in_CIFAR(cifar_path + 'data_batch_3')
X4, y4 = load_in_CIFAR(cifar_path + 'data_batch_4')
X_val, y_val = load_in_CIFAR(cifar_path + 'data_batch_5')

X_test, y_test, dict = load_in_CIFAR(cifar_path + 'test_batch', full_dict=True)
X = X1 + X2 + X3 + X4
y = y1 + y2 + y3 + y4

# Convert data to 0-1 floats, and to categorical.
X = np.array(X).astype('float32')
X /= 255
y = keras.utils.to_categorical(y, num_classes=num_classes)

X_test = np.array(X_test).astype('float32')
X_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

X_val = np.array(X_val).astype('float32')
X_val /= 255
y_val = keras.utils.to_categorical(y_val, num_classes=num_classes)


# Create a test model.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=CIFAR_input_size))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flat parts
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax'))

# Compile mdoel
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


history = model.fit(np.array(X), y, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=1)
print('')
print("Test loss:", score[0])
print("Test accuracy:", score[1])

fig = view_test_data(X_test, y_test)

print(history.history.keys())
# summarize history for accuracy

plt.figure()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
