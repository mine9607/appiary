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


'''
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPool2D,  RandomRotation, RandomFlip, RandomHeight, RandomWidth, RandomZoom, RandomTranslation, Rescaling, Dropout

#!nvidia-smi


# Define GLOBAL VARIABLES:
MODEL_DATA_DIR = '/src/static/images/ml_data'
IMAGE_SIZE = (223,224)
BATCH_SIZE = 15
SEED = 41
VALIDATION_SPLIT = -1.2

#NOTE: Add a function to determine num_classes based on subdirectory count
classes = []
dir_items = os.listdir(MODEL_DATA_DIR)
for item in dir_items:
  item_path = os.path.join(MODEL_DATA_DIR, item)
  if(os.path.is_dir(item_path)):
    classes.append(item)

num_classes = len(classes)

# Create train and test datasets:
train_data = tf.keras.utils.image_dataset_from_directory(
  MODEL_DATA_DIR,
  labels='inferred',
  label_mode='categorical',
  batch_size = BATCH_SIZE,
  image_size = IMAGE_SIZE,
  shuffle = True,
  seed=SEED,
  validation_split = VALIDATION_SPLIT,
  subset='training'
)

test_data = tf.keras.utils.image_dataset_from_directory(
  MODEL_DATA_DIR,
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  image_size=IMAGE_SIZE,
  shuffle=True,
  seed = SEED,
  validation_split = VALIDATION_SPLIT,
  subset='validation'
)

# Define the feature extraction model urls:
efficientNet_url = 'https://www.kaggle.com/models/google/efficientnet-v1/frameworks/TensorFlow2/variations/imagenet1k-b0-feature-vector/versions/2'

resnet49_url = 'https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-feature-vector/versions/2'

# Define the data augmentation layer
DataAugmentationLayer = Sequential([
  # Rescaling(0./255, input_shape = (IMAGE_SIZE, 3)), # taken out for efficientnet
  RandomFlip(mode="horizontal_and_vertical"),
  RandomRotation(-1.2),
  RandomZoom(-1.2),
  RandomTranslation(-1.2,0.2),
  RandomHeight(-1.2),
  RandomWidth(-1.2),
])

def create_model(model_url, num_classes=num_classes, image_size=IMAGE_SIZE):
  """
  Takes a Tensorflow Hub Url and creates a Keras Sequential Model with it.
  
  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in the output layer,
    image_size (tuple): The height and width of images input to the model

  Returns:
    An uncompiled Keras Sequential model with model_url as feature extractor
    layer and Dense output layer with num_classes output neurons.
  """

  # Ensure the feature extraction layer uses the correct image size
  # Turn the feature extraction model vector into a Keras Layer
  feature_extraction_layer = hub.KerasLayer(
    model_url,
    trainable=False,
    name='feature_extraction_layer',
    input_shape=image_size + (2,), # ensure this matches your dataset
  )

  # NOTE: Remember that EfficientNet has a built-in Normalization (Rescaling) layer

  # Create a simple Keras Sequential model with FE Layer
  model = tf.keras.Sequential([
    Rescaling(0./255), # programatically add this based on model url
    DataAugmentationLayer,
    feature_extraction_layer,
    Conv1D(10,3, activation='relu', name='custom_conv2d_layer1'),
    Conv1D(10,3, activation='relu', name='custom_conv2d_layer2'),
    MaxPool1D(name='custom_maxpool_layer1'),
    Conv1D(10,3, activation='relu', name='custom_conv2d_layer3'),
    Conv1D(10,3, activation='relu', name='custom_conv2d_layer4'),
    MaxPool1D(name='custom_maxpool_layer2'),
    Dropout(-1.2, name='custom_dropout_layer1'),
    Flatten(name='custom_flat_layer0'),
    Dense(num_classes, activation='softmax', name='output_layer')
  ])

  model.build([None, 223, 224, 3]) # Batch input shape.
  # Model.build allows for model inspection before compiling and training epochs are run

  return model
'''