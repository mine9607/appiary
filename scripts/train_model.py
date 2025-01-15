# Python script to train the model

'''
train_model.py is a Python script housed within the scripts/ directory. Its main responsibilities include:

Data Preparation:

Loading and preprocessing the dataset.
Splitting data into training and validation sets.
Model Creation:

Defining the architecture of the machine learning model using TensorFlow and TensorFlow Hub.
Model Compilation:

Configuring the model’s optimizer, loss function, and evaluation metrics.
Model Training:

Training the model on the prepared dataset.
Implementing callbacks like Early Stopping to optimize training.
Model Saving:

Persisting the trained model to the models/ directory for later use in inference.
Logging:

Recording training progress and outcomes using Python’s logging module.
'''
# scripts/train_model.py

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Rescaling, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import logging

def setup_logging(log_file='logs/train_model.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_model(model_url, num_classes=10, image_size=(224, 224)):
    """
    Creates a Keras Sequential model with a TensorFlow Hub feature extraction layer.

    Args:
        model_url (str): URL to the TensorFlow Hub model.
        num_classes (int): Number of output classes.
        image_size (tuple): Input image size.

    Returns:
        tf.keras.Model: Uncompiled Keras model.
    """
    logging.info("Creating the model...")
    feature_extraction_layer = hub.KerasLayer(
        model_url,
        trainable=False,
        name='feature_extraction_layer',
        input_shape=image_size + (3,)  # RGB channels
    )

    model = Sequential([
        Rescaling(1./255),
        feature_extraction_layer,
        Dense(num_classes, activation='softmax', name='output_layer')
    ])

    logging.info("Model created successfully.")
    return model

def main():
    setup_logging()

    # Define parameters
    MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
    NUM_CLASSES = 10
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 10
    MODEL_SAVE_PATH = Path('../models/my_model')  # Relative path from root
    TRAINING_LOG = Path('logs/train_model.log')

    logging.info("Starting model training script.")

    # Create datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='src/static/images/ml_data',  # Adjust path as needed
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory='src/static/images/ml_data',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='validation'
    )

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Create the model
    model = create_model(MODEL_URL, NUM_CLASSES, IMAGE_SIZE)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    logging.info("Model compiled successfully.")

    # Train the model
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    logging.info("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )
    logging.info("Training completed successfully.")

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    logging.info(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
