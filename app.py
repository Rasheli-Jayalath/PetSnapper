import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("cat_dog_model.keras")

# Prediction function
def predict(image):
    image = image.resize((100, 100))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 100, 100, 3)
    prediction = model.predict(img_array)[0][0]
    label = "Cat 🐱" if prediction > 0.5 else "Dog 🐶"
    return label

# Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Cat vs Dog Classifier",
    description="Upload an image and find out if it's a cat or a dog!"
)

iface.launch()
