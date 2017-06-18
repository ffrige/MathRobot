# MathRobot
The robot uses a webcam to recognize an algebraic expression and speaks out loud the result.

The first step uses OpenCV to read the webcam streamed frames and find objects in them. I use the findContours and boundingRect functions and remove all objects that are too small or too big compare to the average size. I am assuming all digits to be of similar size.

The second step uses Keras to predict the digit inside each detected object. The model is a convolutional network with two hidden layers trained on the standard MNIST dataset with the addition of the +,-,÷,x signs. The input layer accepts a 28*28*1 shaped array.

The last step uses my daughter's voice to speak out the result of the expression.    


#TODO:
- The CNN model is not completed with operations signs yet.
- The second and last step are not connected with each other yet.
