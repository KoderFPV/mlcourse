import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
from keras import optimizers
from keras.optimizers import Adam
optimizer = Adam()
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
if isinstance(evaluation_result, float):
    test_loss = test_acc = evaluation_result
else:
    test_loss, test_acc = evaluation_result
print(f'Testaccuracy: {test_acc}')
classifications = model.predict(test_images)
print(classifications)
print(test_labels(0))