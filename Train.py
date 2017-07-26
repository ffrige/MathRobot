'''Trains a simple convnet on the MNIST dataset
plus the 4 operation signs (+,-,*,/)

new classes:
+ 10
- 11
* 12
/ 13

Generate samples for the new classes by artificially augmenting sample images

'''

import numpy as np
import math

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adadelta, RMSprop, Adam

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None,expand=False):
    if center is None:
        return image.rotate(angle)
    angle = -angle/180.0*math.pi
    nx,ny = x,y = center
    sx=sy=1.0
    if new_center:
        (nx,ny) = new_center
    if scale:
        (sx,sy) = scale
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=Image.BICUBIC)

batch_size = 128
nb_classes = 14
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)

img_piu = Image.open('signs\piu.png')
img_meno = Image.open('signs\meno.png')
img_per = Image.open('signs\per.png')
img_diviso = Image.open('signs\diviso.png')

###append signs
imgarr_train = []
labelarr_train = []
imgarr_test = []
labelarr_test = []

for i in range(1000):

    #print(img.shape) # NB: this must be 28*28 - grayscale, no RGB

    #apply random transformations to augment data
    angle = np.random.randint(13)-6
    x = 14 + np.random.randint(5)-2
    y = 14 + np.random.randint(5)-2
    sx = np.random.rand()/2+0.5
    sy = np.random.rand()/2+0.5

    img1 = ScaleRotateTranslate(img_piu,angle,(14,14),(x,y),(sx,sy),False)
    img2 = ScaleRotateTranslate(img_meno,angle,(14,14),(x,y),(sx,sy),False)
    img3 = ScaleRotateTranslate(img_per,angle,(14,14),(x,y),(sx,sy),False)
    img4 = ScaleRotateTranslate(img_diviso,angle,(14,14),(x,y),(sx,sy),False)
    
    if i<900:
        imgarr_train.append(img1.getdata())
        imgarr_train.append(img2.getdata())
        imgarr_train.append(img3.getdata())
        imgarr_train.append(img4.getdata())
        labelarr_train.append(10)
        labelarr_train.append(11)
        labelarr_train.append(12)
        labelarr_train.append(13)
    else:
        imgarr_test.append(img1.getdata())
        imgarr_test.append(img2.getdata())
        imgarr_test.append(img3.getdata())
        imgarr_test.append(img4.getdata())
        labelarr_test.append(10)
        labelarr_test.append(11)
        labelarr_test.append(12)
        labelarr_test.append(13)

X_train = np.concatenate((X_train, np.asarray(imgarr_train)), axis=0)
y_train = np.concatenate((y_train, np.asarray(labelarr_train)), axis=0)

X_test = np.concatenate((X_train, np.asarray(imgarr_test)), axis=0)
y_test = np.concatenate((y_train, np.asarray(labelarr_test)), axis=0)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#shuffle training set
p = np.random.permutation(X_train.shape[0])
X_train = X_train[p]
y_train = y_train[p]

#shuffle testing set
p = np.random.permutation(X_test.shape[0])
X_test = X_test[p]
y_test = y_test[p]

#image format theano (samples,width,height,channels)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes,activation='softmax'))

opt = Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('CNN.h5')
#model.save('/output/CNN.h5')
