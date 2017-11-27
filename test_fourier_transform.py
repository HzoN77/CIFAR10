from psychopy import visual
from psychopy.visual import filters

import cv2
import numpy as np
from matplotlib import pyplot as plt

#from scipy import ndimage, misc


img = cv2.imread('messi.jpg', 0)
img = np.array(img).astype('float32')
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.figure()
plt.subplot(321), plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])


lp_filt = filters.butter2d_lp(
    size=np.shape(img),
    cutoff=0.80,
    n=10
)

filtered_img_freq = np.multiply(fshift, lp_filt)
filtered_img_mag = np.abs(filtered_img_freq)
plt.subplot(323), plt.imshow(np.log(1+filtered_img_mag), cmap = 'gray')
plt.title('LP-Filtered magnitude'), plt.xticks([]), plt.yticks([])

rec_img = np.abs(np.fft.ifft2(filtered_img_freq))
rec_img = (rec_img - np.min(rec_img)) / (np.max(rec_img)-np.min(rec_img))

plt.subplot(324),plt.imshow(rec_img, cmap = 'gray')
plt.title('LP-Reconstructed'), plt.xticks([]), plt.yticks([])


hp_filt = filters.butter2d_hp(
    size=np.shape(img),
    cutoff=0.3,
    n=10
)

filtered_img_freq = np.multiply(fshift, hp_filt)
filtered_img_mag = np.abs(filtered_img_freq)
plt.subplot(325), plt.imshow(np.log(1+filtered_img_mag), cmap = 'gray')
plt.title('HP-Filtered magnitude'), plt.xticks([]), plt.yticks([])

rec_img = np.abs(np.fft.ifft2(filtered_img_freq))
rec_img = (rec_img - np.min(rec_img)) / (np.max(rec_img)-np.min(rec_img))

plt.subplot(326),plt.imshow(rec_img, cmap = 'gray')
plt.title('HP reconstructed'), plt.xticks([]), plt.yticks([])
plt.savefig('test.pdf')
plt.savefig('test.png')
plt.show()

''' 
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

img = cv2.imread('messi.jpg', 0)
img = np.array(img).astype('float32')
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.figure()
plt.subplot(221), plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# A slightly "wider", but sill very simple highpass filter
kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])

kernel = np.ones((5,5)) / 25
print(kernel)

highpass_5x5 = ndimage.convolve(img, kernel)
#highpass_5x5 *= (255.0/highpass_5x5.max())
highpass_5x5 += np.max(highpass_5x5)
plt.subplot(223),plt.imshow(highpass_5x5, cmap = 'gray')
plt.title( 'Simple 5x5 Highpass')

f = np.fft.fft2(highpass_5x5)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(224),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum 5x5 Lowpass'), plt.xticks([]), plt.yticks([])


plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2


def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
plot.i = 0

# Load the data...
im = cv2.imread('messi.jpg', cv2.IMREAD_COLOR)
data = np.array(im, dtype='float32')
plot(data, 'Original')

# A very simple and very narrow highpass filter
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
kernel = np.reshape(kernel, (3,3,1))
highpass_3x3 = ndimage.convolve(data, kernel)
plot(highpass_3x3, 'Simple 3x3 Highpass')

# A slightly "wider", but sill very simple highpass filter
kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
kernel = np.reshape(kernel, (5,5,1))
highpass_5x5 = ndimage.convolve(data, kernel)
plot(highpass_5x5, 'Simple 5x5 Highpass')

# Another way of making a highpass filter is to simply subtract a lowpass
# filtered image from the original. Here, we'll use a simple gaussian filter
# to "blur" (i.e. a lowpass filter) the original.
lowpass = ndimage.gaussian_filter(data, 3)
gauss_highpass = data - lowpass
plot(gauss_highpass, r'Gaussian Highpass, $\sigma = 3 pixels$')

plt.show()
'''
