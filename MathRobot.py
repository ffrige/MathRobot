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

    #turn it gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,gray = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

    """
    #Use fixed image when camera is not available
    img = cv2.imread('tests/test6.png',1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """
    
    #find bouding boxes for objects
    rect = OD.boundingBoxes(gray,100,100)

    #predict digit for each object 
    digits = OD.predictDigits(rect,gray)
    
    #print predicted digits and bounding boxes
    for i in range(len(digits)):
        cv2.putText(img,OD.DigitToString(digits[i][0]),(digits[i][1],digits[i][2]+10), 0, 2,color=(10,255,10),thickness=2)
    
    for i in range(len(rect)):
        cv2.rectangle(img,(rect[i][0],rect[i][1]),(rect[i][0]+rect[i][2],rect[i][1]+rect[i][3]),(0,255,255),2)
    
    #show original image
    cv2.imshow("Webcam View", img)

    #speak out result of expression
    SO.speak(digits)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
