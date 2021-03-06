"""
This module provides two functions: boundingBoxes and predictDigits

- boundingBoxes: reads an input gray image and returns a list of rectangles that
    represent the bouding boxes to all objects in the image. I am assuming
    that the image has a white backgroud (e.g. sheet of paper) with black
    digits written on it. Objects that are too large or too small are removed
    and objects that overlap others are also eliminated.

- predictDigits: reads the same input gray image and the bounding boxes found
    by the previous function. Runs a classifier over the regions and outputs
    a list of digit, with their regions coordinates. Note that not all the input
    regions make it to the output, because they are screened for containing
    actual digits.


"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import keras
from keras.models import load_model
model = load_model('CNN.h5')


def boundingBoxes(grayImg,filterA_value,filterB_value):
    #load image, convert to gray, blur and detect edges
    gray = cv2.bilateralFilter(grayImg, 9, 100, 100)
    edges = cv2.Canny(gray, 30, 200)
    #find contours in image
    (_, cnts, _) = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #draw rectangles around them
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    avg_width = 0
    avg_height = 0
    rect = []
    #find all bounding boxes
    for i in range(len(cnts)):
        testRect = cv2.boundingRect(cnts[i])
        #ignore objects that are too small or too big
        if (testRect[2] * testRect[3]) < 10 or (testRect[2] * testRect[3]) > 10000:
            continue
        rect.append(testRect)
        rectIdx = len(rect)-1
        avg_width = (avg_width*rectIdx + rect[rectIdx][2])/(rectIdx+1)
        avg_height = (avg_height*rectIdx + rect[rectIdx][3])/(rectIdx+1)
    #print("Found {0} objects with avg_width {1} and avg_height {2}".format(len(cnts),avg_width,avg_height))

    #remove overlapping rectangles, keep larger ones only
    toRemove = []
    for i in range(len(rect)):
        for j in range(i+1,len(rect)):
            distance = ((rect[i][0]-rect[j][0])**2+(rect[i][1]-rect[j][1])**2)**.5
            if distance<rect[i][2]/2 or distance<rect[i][3]/2 or distance<rect[j][2]/2 or distance<rect[j][3]/2:
                area_i = rect[i][2] * rect[i][3]
                area_j = rect[j][2] * rect[j][3]
                if area_i < area_j:
                    toRemove.append(i)
                else:
                    toRemove.append(j)                
    rect = np.delete(rect,toRemove,0)
    return(rect)


def predictDigits(rect,grayImg):
    digits = []
    #convert rectangles into squares of 28*28        
    for i in range(len(rect)):
        y = rect[i][1]
        h = rect[i][3]
        x = rect[i][0]
        w = rect[i][2]

        #add some padding to increase reading area
        pad_x = min(int(w*0.2),x,grayImg.shape[1]-x-w)
        pad_y = min(int(h*0.2),y,grayImg.shape[0]-y-h)
        square = grayImg[y-pad_y:y+h+pad_y,x-pad_x:x+w+pad_x]

        h = square.shape[0]
        w = square.shape[1]
        delta = int(abs(h-w)/2)

        #add more padding to make it square
        pad = int(min(h,w)*0.1)
        if w>h:
            square = cv2.copyMakeBorder(square,delta+pad,delta+pad,pad,pad,cv2.BORDER_CONSTANT,value=255)
        else:
            square = cv2.copyMakeBorder(square,pad,pad,delta+pad,delta+pad,cv2.BORDER_CONSTANT,value=255)

        #add blurring here before resizing
        #otherwise details will be lost when resizing
        square = cv2.GaussianBlur(square,(5,5),5)

        #resize to 28*28 pixel size
        square = cv2.resize(square,(28,28))
        
        #run classifier over each square
        imgarr = square.reshape(1,28,28,1) #for Conv net
        imgarr = imgarr.astype('float32')
        imgarr /= 255
        imgarr = (1 - imgarr)
        prediction = model.predict(imgarr,verbose=0)
        #print(prediction)
        if np.amax(prediction) > 0.3:
            digits.append([np.argmax(prediction),x,y,w,h])
        #plt.imshow(imgarr.reshape(28,28), cmap='Greys', interpolation='nearest')
        #plt.show()
    return digits


def DigitToString(in_digit):
        if in_digit<10:
            return str(in_digit)
        if in_digit == 10:
            return '+'
        if in_digit == 11:
            return '-'
        if in_digit == 12:
            return 'x'
        if in_digit == 13:
            return '/'

