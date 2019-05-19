from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
----------------------------------------------------------------------------------------
Meant to preprocess the data. We inspect the image ourselves to see the range of
values the test images will have. We see that their pixel values are 0 to 255 .
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()
'''
-----------------------------------------------------------------------------------------
'''

'''
----------------------------------------------------------------------------------------
We need to scale the image and test image values to 0 to 1 so we divide by 255
which is the maximum value of pixels.
'''
train_images = train_images / 255.0
test_images = test_images / 255.0
'''
----------------------------------------------------------------------------------------
'''

#Displaying the first 25 images in the training set and displaying the class
#name below it. We can see if the classification is correct.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

'''
Setting up the layers which extract different "levels" of representations of the
data fed in. Lower layers deal with more miniscule details than higher levels.
1. First play tf.keras.layers.Flattern transforms the format of the images of
a 2D array 28 x 28 pixels to a 1D array of 28 & 28 = 784 pixels. In the first
layer, we're unpacking the rows of pixels and lining them up. Layer only deals with
reformatting th data.
2. We're utilizing two tf.keras.layers.Dense layers. These are "densely" connected
or "fully-connected" neural layers. First "Dense" layer has 128 nodes/neurons
and the last layer has 10 softmax nodes which returns an array of 10 probability
scores that sum to 1. Each node contains a score that indiciates the probability
the current image belongs to one of these 10 classes --> There are 10 clothing items
to categorize so each soft node represents a % chance the specific image is that one.
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

'''
Need to compile the model before the model's ready for training.
--> Optimizier is how the model is upated based on the data it sees and its loss function
--> loss - the loss function - is the measure of how accurate the model is during
training. We want to minimize this function to converge the model onto a accurate value
--> metrics is used to monitor the training and training steps. This one is
utilizing accuracy as the metric.
All Optimizers: https://keras.io/optimizers/

'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
