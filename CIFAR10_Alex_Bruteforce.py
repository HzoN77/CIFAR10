import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import cv2
from CIFAR_help_functions import *

# Hyper params
batch_size = 128
num_classes = 10
epochs = 1
CIFAR_input_size = (32, 32, 1)

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
X = np.array(X1).astype('float32')
X /= 255
y = keras.utils.to_categorical(y1, num_classes=num_classes)

X_test = np.array(X_test).astype('float32')
X_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

X_val = np.array(X_val).astype('float32')
X_val /= 255
y_val = keras.utils.to_categorical(y_val, num_classes=num_classes)

# Convert to grayScale.
X = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in X]
X_test = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x  in X_test]
X_val = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x  in X_val]

# Doing a full 101 laps with 0, 0.01, 0.02 ... 1.0
test_case_name = 'CIFAR_brute_force_test_LP'
f = np.linspace(0.001, 0.3, num=100)
progress_file = open(test_case_name + '/results.txt', 'w')
for i in range(len(f)):
    ## Manipulate data
    lp_filter = filters.butter2d_hp(size=(32, 32), cutoff=0.11, n=10)
    X_gray = filter_data(X, lp_filter)
    X_test_gray = filter_data(X_test, lp_filter)
    X_val_gray = filter_data(X_val, lp_filter)

    X_gray = np.expand_dims(X_gray, axis=-1)
    X_test_gray = np.expand_dims(X_test_gray, axis=-1)
    X_val_gray = np.expand_dims(X_val_gray, axis=-1)

    ## Create model
    alexnet = create_AlexNet(network_input_shape=CIFAR_input_size, num_classes=num_classes)

    ## Train model
    hist = alexnet.fit(X_gray, y, batch_size=batch_size, epochs=epochs, validation_data=(X_val_gray, y_val))
    score = alexnet.evaluate(X_test_gray, y_test)

    ## Save results
    fullname = test_case_name + "/LP-Freq_" + str(f[i])
    plt.clf()
    plt.subplot(211)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    # summarize history for loss
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fullname + '.png')

    progress_file.write("Run:" + str(i) + "; Freq:" + str(f[i]) + "; results:" + '; '.join([str(x) for x in score]) + '\n')

progress_file.close()