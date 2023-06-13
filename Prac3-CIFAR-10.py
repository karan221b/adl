!pip install tensorflow

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# Convert labels to one-hot encoding format y_train =
keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# Define the model architecture
model = keras.Sequential([
  keras.Input(shape=(32, 32, 3)),
  layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Flatten(),
  layers.Dropout(0.5),
  layers.Dense(10, activation="softmax"),
])
# Compile the model model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
# Train the model model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_test,y_test))
# Save the trained model to a file
model.save("cifar10_model.h5")

import numpy as np 
from PIL import Image
# Load the saved model
model = keras.models.load_model("cifar10_model.h5")
# Load and preprocess the test image
img = Image.open("cat.jpg")
img = img.resize((32, 32))
img_array = np.array(img)
img_array = img_array.astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)
# Make predictions on the test image 
predictions = model.predict(img_array)# Get the predicted class label 
class_label = np.argmax(predictions)
# Print the predicted class label 
print("Predicted class label:", class_label)
