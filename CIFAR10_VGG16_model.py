#
#
# Initial tests on CIFAR10 data. how to read/load data and handle it.

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
import matplotlib.pyplot as plt

from CIFAR_help_functions import *
from cracked_baffoon import *




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

fourier_test(cv2.cvtColor(X[45], cv2.COLOR_RGB2GRAY), 0.01, 0.02)

vggModel = create_fake_VGG16(CIFAR_input_size, num_classes)
vggModel.summary()

# Compile model

model = create_AlexNet(num_classes=num_classes)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

# #model.fit(np.array(X), y, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
#
#
#
# #score = model.evaluate(X_test, y_test, verbose=1)
# #print('')
# #print("Test loss:", score[0])
# #print("Test accuracy:", score[1])
#
# fig = view_test_data(X_test, y_test)
# plt.show()



