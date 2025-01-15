from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub

def load_model():
    model_path = Path(__file__).parent/"models"
    if model_path.is_dir():
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        print("Model loaded successfully.")
        return model
    else:
        print(f"Model directory {model_path} does not exist.")
    return None