# MathRobot
The robot uses a webcam to recognize an algebraic expression and speaks out loud the result.

It is meant to work by showing a piece of paper with some digits written on it, (e.g.
15-8) to simplify the objects recognition.

The first step uses OpenCV to read the webcam streamed frames and find objects in them. I use the findContours and boundingRect functions to find all objects, then remove those that are too small or too big compared to the average size (I am assuming all digits to be of similar size).

The second step uses Keras to predict the digit inside each detected region. The model is a convolutional network with two hidden layers trained on the standard MNIST dataset with the addition of the +,-,÷,x signs. The input layer accepts a 28*28*1 shaped array. If a digit is found with probability higher than 0.5 than it is added to the list.

The last step uses my daughter's voice to speak out the result of the expression. Note: my daughter speaks Chinese...


#TODO:
- The CNN model is not completed with operations signs yet.

