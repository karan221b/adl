import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the VGG16 model with pre-trained weights
model = VGG16()

# Load the image to detect objects in
image = load_img('final.jpg', target_size=(224, 224))

# Convert the image to a numpy array
image = img_to_array(image)

# Reshape the image data for VGG
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))


# Preprocess the image
image = preprocess_input(image)

# Make predictions on the image using the VGG model
predictions = model.predict(image)


# Decode the predictions
decoded_predictions = decode_predictions(predictions, top=2)


# Print the predictions with their probabilities
for i, prediction in enumerate(decoded_predictions[0]):
  print("Object ", i+1, ": ", prediction[1], ", Probability: ", prediction[2])
