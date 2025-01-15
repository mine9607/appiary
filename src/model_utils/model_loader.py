from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from config import MODEL_PATH

model = None

def load_model():
    global model

    model_path = MODEL_PATH
    if model_path.is_dir():
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        print("Model loaded successfully.")
    else:
        print(f"Model directory {model_path} does not exist.")
    return model