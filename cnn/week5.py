import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
import tensorflow as tf
from keras.backend import set_session


# Tell Tensorflow to only use as much GPU memory
# it needs (not all of it)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Hardcoded image dimensions
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

SAMPLE_SIZE = 600

# Hardcoded number of classes (digits from 0 to 9)
NUM_CLASSES = 10

# Flip the test and train to get bigger test set
(x_test, y_test), (x_train, y_train) = mnist.load_data()

# Reduce sample size to achieve overfitting
x_train = x_train[:SAMPLE_SIZE] / 255.0
x_test = x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = np.eye(NUM_CLASSES)[y_train[:SAMPLE_SIZE].astype(np.int32)]
y_test = np.eye(NUM_CLASSES)[y_test.astype(np.int32)]

# Create the Keras model and start adding the layers.
model = Sequential()
model.add(Conv2D(filters=8,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 activation='tanh',
                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=16,
                 kernel_size=(3, 3),
                 activation='tanh'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=NUM_CLASSES, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Increased epochs to achieve overfitting
model.fit(x=x_train, y=y_train, epochs=30)

loss, accuracy = model.evaluate(x=x_test, y=y_test)
print("Accuracy in test set: %.2f" % accuracy)
print("Lost in test set: %.2f" % loss)
