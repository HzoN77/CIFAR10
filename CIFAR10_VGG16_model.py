#
#
# Initial tests on CIFAR10 data. how to read/load data and handle it.

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

from CIFAR_help_functions import *


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

model.summary()

# Compile mdoel

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


history = model.fit(np.array(X), y, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=1)

print('')
print("Test loss:", score[0])
print("Test accuracy:", score[1])

fig = view_test_data(X_test, y_test, model)

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