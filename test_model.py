import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# set model path and test image(resized to 224, 224 for mobilenet)
MODEL_PATH = "vegetable_identifier_model.h5"
IMAGE_PATH = "testing_images/banana.jpg"  # Replace with the image you want to test
IMAGE_SIZE = (224, 224)

# the load_model function from TF - used to load the model
model = load_model(MODEL_PATH)

# load the class indices from the JSON file
with open("class_indices.json", 'r') as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# load image into TF
img = image.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize same as training

# make a prediction via the model
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_label = class_names[predicted_index]
confidence = predictions[0][predicted_index]

# output the result as a prediction with a confidence rating in percentage.
print(f"🔍 Predicted: {predicted_label} ({confidence * 100:.2f}% confidence)")
