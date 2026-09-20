# step 1
# pull pokemon images out of files

import os
from PIL import Image
from pathlib import Path
import random 
import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

random.seed(42)

# step 2
# split data into training and testing sets labelled, split 80/20 randomized per pokemon folder

# parse through PokemonDataResampled
# for each folder, randomly select 80% of the images and create x, y pairs using the folder name. this is the train set. 
# the compliment of this set of x, y pairs becomes the test set

train_data = []
test_data = []

pokemon_dataset = Path("PokemonDataResampled")

for folder in pokemon_dataset.iterdir():
    if not folder.is_dir():
        continue

    label = folder.name

    images = [p for ext in ("*.png", "*.jpg", "*.jpeg") for p in folder.glob(ext)]
    random.shuffle(images)

    split = int(len(images) * 0.8)

    train_images = images[:split]
    test_images = images[split:]

    for image in train_images:
        train_data.append((image, label))
    
    for image in test_images:
        test_data.append((image, label))

X_train = [str(path) for path, label in train_data]
y_train = [pokemon_to_id[label] for path, label in train_data]

X_test = [str(path) for path, label in test_data]
y_test = [pokemon_to_id[label] for path, label in test_data]

y_train = np.array(y_train)
y_test = np.array(y_test)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, (500, 500))
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

# X_train, y_train = zip(*train_data)
# X_test, y_test = zip(*test_data)

# X_train = np.array([
#     np.array(Image.open(path).convert("RGB"))
#     for path in X_train
# ])

# X_test = np.array([
#     np.array(Image.open(path).convert("RGB"))
#     for path in X_test
# ])

# X_train = X_train / 255
# X_test = X_test / 255

# pokemon_names = sorted(set(y_train))

# pokemon_to_id = {
#     name: i for i , name in enumerate(pokemon_names)
# }

# y_train = np.array([pokemon_to_id[name] for name in y_train])
# y_test = np.array([pokemon_to_id[name] for name in y_test])

# y_train = to_categorical(y_train, 150)
# y_test = to_categorical(y_test, 150)

# step 3
# define cnn model using keras 

input_layer = tf.keras.layers.Input(shape=(500, 500, 3))

model = tf.keras.Sequential([
    input_layer, 
    layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(), 
    layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(), 
    layers.Flatten(),
    layers.Dense(150, activation='softmax')
])

# create the layers of the cnn where the input is the 500 by 500 image, the middle layers consist of conv2d and maxpooling2d layers
# and the output is a softmax of pokemon classes

# step 4 
# train model and graph loss over epochs

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model by defining batch size, epochs, learning rate, optimizer

history = model.fit(
    X_train, 
    y_train, 
    validation_split=0.2, 
    epochs=10, batch_size=64
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# step 5
# test accuracy of model on testing data

# test data on the remaining 20% of data as well as some out of sample pokemon found on google images 