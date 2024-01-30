import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Flatten

data = tf.keras.datasets.fashion_mnist

(trainig_images, training_labels), (test_images, test_labels) = data.load_data()

trainig_images = trainig_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(trainig_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)