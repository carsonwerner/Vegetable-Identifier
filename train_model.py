# Carson Werner
# Final Project
# For this project, I will attempt to train a model using MobileNetV2, or maybe FastAI or a similar library to do the heavy lifting
# and non-practical stuff. I will train on top of that and use the model to identify vegetables in images.

import subprocess
import sys
import os
import importlib

def install_dependencies(packages):
    """
    Installs a list of Python librariews using pip install if they are not already installed
    """
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['tensorflow', 'json', 'matplotlib', 'pandas']
install_dependencies(required_packages)



import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

#set directory
BASE_DIR = "vegetable_images"
#number of pics the model sees at a time
BATCH_SIZE = 32
#resize for mobilenet. refer to for size info https://huggingface.co/docs/transformers/v4.39.2/en/model_doc/mobilenet_v2
IMAGE_SIZE = (224, 224)
#amount of passes through the full training dataset. refer to https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model
EPOCHS = 10

# from what i understand, this essentially controls how fast previous info is overwritten each epoch(pass throug the dataset)
LEARNING_RATE = 1e-4

# data prep. split the images into train, test, and validation sets
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "validation")
test_dir = os.path.join(BASE_DIR, "test")


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir,
                                              target_size=IMAGE_SIZE,
                                              batch_size=BATCH_SIZE,
                                              class_mode="categorical")

val_gen = val_test_datagen.flow_from_directory(val_dir,
                                               target_size=IMAGE_SIZE,
                                               batch_size=BATCH_SIZE,
                                               class_mode="categorical")

test_gen = val_test_datagen.flow_from_directory(test_dir,
                                                target_size=IMAGE_SIZE,
                                                batch_size=BATCH_SIZE,
                                                class_mode="categorical",
                                                shuffle=False)

# build the first model using mobilenetv2
base_model = MobileNetV2(include_top=False,
                         input_shape=(*IMAGE_SIZE, 3),
                         weights='imagenet') 
# check https://www.image-net.org/ for info on imagenet

base_model.trainable = False  # lock model from learning new data. it will build upon this.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model using dataset and validation set
checkpoint = ModelCheckpoint("vegetable_identifier_model.h5",
                             save_best_only=True,
                             monitor='val_accuracy',
                             mode='max')

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS,
                    callbacks=[checkpoint])

# === Evaluate on Test Set ===
loss, accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# === Save class label mapping ===
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved class_indices.json")
