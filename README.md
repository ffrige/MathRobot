# MathRobot
The robot uses a webcam to recognize an algebraic expression and speaks out loud the result.

It is meant to work by showing a piece of paper with some digits written on it, (e.g.
15-8) to simplify the objects recognition.

![examples of digits detection](/result4.png)
![examples of digits detection](/result6.png)

The first step uses OpenCV to read the webcam streamed frames and find objects in them. I use the findContours and boundingRect functions to find all objects, then remove those that are too small or too big (I am assuming all digits to be of similar size).

The objects are padded and rescaled to 28x28 squares to fit the input of the neural network.

The second step uses a convolutional network with two hidden layers to predict the digit inside each detected region. The model is trained on the standard MNIST dataset with the addition of the +,-,x,÷ signs. The input layer accepts a 28x28x1 shaped array. If a digit is found with probability higher than 0.3 than it is added to the list.

The addition of the +,-,x,÷ signs was done via artificial augmentation with random affine transformations (translations, rotations, scalings) in the PIL library. A total of 1000 images for each sign was added to the database, 90% for training and 10% for testing. A cross-validation subset is automatically extracted from the train set by Keras.

The final step uses my daughter's voice to speak out the result of the expression. Note: my daughter speaks Chinese...
