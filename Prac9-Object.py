import numpy as np 
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the VGG16 model with pre-trained weights
model = VGG16()


# Load the image to detect objects in
img = load_img('image.jpg', target_size=(224, 224))


# Convert the image to a numpy array
img_arr = img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr = preprocess_input(img_arr)


# Predict the objects in the image
preds = model.predict(img_arr)
decoded_preds = decode_predictions(preds, top=5)[0]


# Print the predicted objects and their probabilities
for pred in decoded_preds:
  print(f"{pred[1]}: {pred[2]*100:.2f}%")
