import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Path to model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")

def load_model():
    """
    Load trained deepfake detection model (Keras .h5).
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    model.make_predict_function()
    return model

def preprocess_image_for_model(image_path, target_size=(224, 224)):
    """
    Preprocess uploaded image for prediction:
    - Resize to target_size
    - Normalize pixel values to [0,1]
    - Add batch dimension
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, C)
    return arr

def predict_image(model, image_path, threshold=0.5):
    """
    Run prediction on image:
    Returns (label, confidence).
    Assumes model outputs either:
    - sigmoid (1 neuron): prob_fake
    - softmax (2 neurons): [prob_real, prob_fake]
    """
    x = preprocess_image_for_model(image_path)
    preds = model.predict(x)

    # Handle sigmoid or softmax
    if preds.ndim == 2 and preds.shape[1] == 2:
        prob_fake = float(preds[0, 1])
    else:
        prob_fake = float(preds[0])  # assume sigmoid

    if prob_fake >= threshold:
        label = "Fake"
        confidence = round(prob_fake * 100, 2)
    else:
        label = "Real"
        confidence = round((1 - prob_fake) * 100, 2)

    return label, confidence
