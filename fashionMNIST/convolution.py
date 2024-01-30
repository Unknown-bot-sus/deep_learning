import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.nn import relu, softmax

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (testing_images, testing_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1) # chaning gray scale to rgb shape
training_images = training_images / 255.0 # normalizing 

testing_images = testing_images.reshape(10000, 28, 28, 1) # converting gray scale to rgb shape
testing_images = testing_images / 255.0 # normalizing

model = Sequential([
    # number of neurons, grid size
    Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation=relu),
    Dense(10, activation=softmax)
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)

model.evaluate(testing_images, testing_labels)

model.summary