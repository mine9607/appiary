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

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, RandomRotation, RandomFlip, RandomHeight, RandomWidth, RandomZoom, RandomTranslation, Dropout, RandomTranslation, Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from config import MODEL_PATH, MODEL_DATA_DIR, RESNET_MODEL_URL, IMAGE_SIZE
from pathlib import Path
import logging

# Funtion to setup logging during model training
def setup_logging(log_file='logs/train_model.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Function to determine the classes from the MODEL_DATA_DIR folder
def get_classes():
  ''' Determine the classes from subdirectory structure'''
  classes = []
  dir_items = Path(MODEL_DATA_DIR).iterdir()
  for item in dir_items:
    if item.is_dir():
      classes.append(item.name)

  return classes

# Function to create the model architecture
def create_model(num_classes, model_url=RESNET_MODEL_URL, image_size=IMAGE_SIZE):
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
    FeatureExtractionLayer = hub.KerasLayer(
        model_url,
        trainable=False,
        name='feature_extraction_layer',
        input_shape=image_size + (3,)  # RGB channels
    )

    DataAugmentationLayer = Sequential([
      RandomFlip(mode="horizontal_and_vertical"),
      RandomRotation(0.2),
      RandomZoom(0.2),
      RandomTranslation(0.2,0.2),
      RandomHeight(0.2),
      RandomWidth(0.2),
    ])

    model = Sequential([
        Rescaling(1./255),
        DataAugmentationLayer,
        FeatureExtractionLayer,
        # Conv2D(num_classes, 3, activation='relu', name='custom_conv2d_layer1'),
        # Conv2D(num_classes, 3, activation='relu', name='custom_conv2d_layer2'),
        # MaxPool2D(name='custom_pool_layer1'),
        # Flatten(name='custom_flat_layer1'),
        Dense(num_classes, activation='softmax', name='output_layer')
    ])

    logging.info("Model created successfully.")
    return model

def main():
  setup_logging()

  # Check for GPU
  physical_devices = tf.config.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    print("GPU is available.")
  else:
    print("GPU is not available.")

  # Define Model parameters
  NUM_CLASSES = len(get_classes())
  BATCH_SIZE = 4, 
  EPOCHS = 10
  MODEL_SAVE_PATH = MODEL_PATH  
  TRAINING_LOG = Path('logs/train_model.log')
  FEATURE_MODEL_URL = RESNET_MODEL_URL

  logging.info("Starting model training script.")

  # Create datasets from MODEL_DATA_DIR
  train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=MODEL_DATA_DIR,  
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
    directory=MODEL_DATA_DIR,
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
  model = create_model(NUM_CLASSES,FEATURE_MODEL_URL, IMAGE_SIZE)

  # Compile the model
  model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )
  
  logging.info("Model compiled successfully.")

  # Define an EarlyStopping Callback for the loss function 
  early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
  
  logging.info("Starting training...")
  
  # Train the model
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