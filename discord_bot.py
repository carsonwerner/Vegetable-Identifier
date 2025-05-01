# discord_bot/discord_bot.py

import os
import json
import discord
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


MODEL_PATH = "vegetable_identifier_model.h5"
LABEL_PATH = "class_indices.json"
IMAGE_SIZE = (224, 224)

#model and map(check desktop folder cwerner)
model = load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# preprocess(resize, converts to numpy, normalize pixel values 0-1)
def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# discord bot setup
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

#discord wrapper for bot initializiation 
#https://stackoverflow.com/questions/68215349/how-can-i-make-a-wrapper-function-for-discord-py-commands
@client.event
async def on_ready():
    print(f"Bot connected as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['jpg', 'jpeg', 'png']):
                file_path = f"temp_{attachment.filename}"
                await attachment.save(file_path)

                try:
                    img_array = preprocess_image(file_path)
                    predictions = model.predict(img_array)
                    pred_idx = np.argmax(predictions[0])
                    pred_label = class_names[pred_idx]
                    confidence = predictions[0][pred_idx]

                    await message.channel.send(
                        f"**Prediction:** `{pred_label}`\n📊 **Confidence:** `{confidence * 100:.2f}%`"
                    )
                except Exception as e:
                    await message.channel.send(f"Failed to predict: {e}")
                finally:
                    os.remove(file_path)

# DISCORD BOT TOKEN
TOKEN = "" # bot token redacted for security reasons
client.run(TOKEN)
