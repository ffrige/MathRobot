"""
This program reads the input frames from a webcam, looks for a mathematical
expression, evaluates it and speaks out the result.
"""


import ObjDetect as OD
import SpeakOut as SO
import cv2

cap = cv2.VideoCapture(0)
assert cap.isOpened(),"Camera not found!"

while(True):
    #load image
    ret, img = cap.read()
    #cv2.imshow("View", img)
    #img = cv2.imread('test.png',1)

    #turn it gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find bouding boxes for objects
    rect = OD.boundingBoxes(gray)

    #predict digit for each object 
    digits = OD.predictDigits(rect,gray)

    #print predicted digits and bounding boxes
    for i in range(len(digits)):
        cv2.putText(img,str(digits[i][0]),(digits[i][1],digits[i][2]), 0, 2,color=(0,255,255),thickness=3)
        cv2.rectangle(img,(digits[i][1],digits[i][2]),(digits[i][1]+digits[i][3],digits[i][2]+digits[i][4]),(0,255,255),3)

    #show original image
    cv2.imshow("Webcam View", img)

    #speak out result of expression
    SO.speak(digits)

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
