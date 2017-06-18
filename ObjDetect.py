import numpy as np
import cv2
import matplotlib.pyplot as plt

import keras
from keras.models import load_model
model = load_model('CNN.h5')


def boundingBoxes(gray):
    #load image, convert to gray, blur and detect edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
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
        #only consider objects that are sized at least 10% of full image
        if testRect[2] < 64 or testRect[3] < 48:
            continue
        rect.append(testRect)
        rectIdx = len(rect)-1
        avg_width = (avg_width*rectIdx + rect[rectIdx][2])/(rectIdx+1)
        avg_height = (avg_height*rectIdx + rect[rectIdx][3])/(rectIdx+1)
    #print("Found {0} objects with avg_width {1} and avg_height {2}".format(len(cnts),avg_width,avg_height))

    #remove objects that are too small or too big
    toRemove = []
    trsh = 0.2
    for i in range(len(rect)):
        if rect[i][2]<avg_width*trsh or rect[i][2]>avg_width/trsh or rect[i][3]<avg_height*trsh or rect[i][3]>avg_height/trsh:
            toRemove.append(i)
    rect = np.delete(rect,toRemove,0)

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

def predictDigits(rect,gray):
    digits = []
    #convert rectangles into squares of 28*28        
    for i in range(len(rect)):
        y = rect[i][1]
        h = rect[i][3]
        x = rect[i][0]
        w = rect[i][2]
        pad_x = min(int(w*0.2),x,gray.shape[1]-x-w)
        pad_y = min(int(h*0.2),y,gray.shape[0]-y-h)
        #square = gray[y:y+h,x:x+w]
        square = gray[y-pad_y:y+h+pad_y,x-pad_x:x+w+pad_x]

        h = square.shape[0]
        w = square.shape[1]
        delta = int(abs(h-w)/2)

        #add padding to make it square
        pad = int(min(h,w)*0.1)
        if w>h:
            square = cv2.copyMakeBorder(square,delta+pad,delta+pad,pad,pad,cv2.BORDER_CONSTANT,value=255)
        else:
            square = cv2.copyMakeBorder(square,pad,pad,delta+pad,delta+pad,cv2.BORDER_CONSTANT,value=255)
        square = cv2.resize(square,(28,28))
        
        #run classifier over each square
        #imgarr = square.reshape(1,784) #for FF net
        imgarr = square.reshape(1,28,28,1) #for Conv net
        imgarr = imgarr.astype('float32')
        imgarr /= 255
        imgarr = (1 - imgarr)
        prediction = model.predict(imgarr,verbose=0)
        if np.amax(prediction) > 0.5:
            digits.append([np.argmax(prediction),x,y])
        #plt.imshow(imgarr.reshape(28,28), cmap='Greys', interpolation='nearest')
        #plt.show()
    return digits



cap = cv2.VideoCapture(0)
assert cap.isOpened(),"Camera not found!"

#img = cv2.imread('test.png',1)

while(True):
    #load image
    ret, img = cap.read()
    #cv2.imshow("View", img)

    #turn it gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find bouding boxes for objects
    rect = boundingBoxes(gray)

    #predict digit for each object 
    digits = predictDigits(rect,gray)

    #print predicted digits and bounding boxes
    for i in range(len(digits)):
        cv2.rectangle(img,(rect[i][0],rect[i][1]),(rect[i][0]+rect[i][2],rect[i][1]+rect[i][3]),(0,255,255),3)
        cv2.putText(img,str(digits[i][0]),(digits[i][1],digits[i][2]), 0, 2,color=(0,255,255),thickness=3)

    #show original image
    cv2.imshow("Webcam View", img)

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
