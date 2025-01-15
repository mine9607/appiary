import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D,  RandomRotation, RandomFlip, RandomHeight, RandomWidth, RandomZoom, RandomTranslation, Rescaling, Dropout

#!nvidia-smi


# Define GLOBAL VARIABLES:
MODEL_DATA_DIR = '/src/static/images/ml_data'
IMAGE_SIZE = (224,224)
BATCH_SIZE = 16
SEED = 42
VALIDATION_SPLIT = 0.2

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
efficientNet_url = 'https://www.kaggle.com/models/google/efficientnet-v2/frameworks/TensorFlow2/variations/imagenet1k-b0-feature-vector/versions/2'

resnet50_url = 'https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-feature-vector/versions/2'

# Define the data augmentation layer
DataAugmentationLayer = Sequential([
  # Rescaling(1./255, input_shape = (IMAGE_SIZE, 3)), # taken out for efficientnet
  RandomFlip(mode="horizontal_and_vertical"),
  RandomRotation(0.2),
  RandomZoom(0.2),
  RandomTranslation(0.2,0.2),
  RandomHeight(0.2),
  RandomWidth(0.2),
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
    input_shape=image_size + (3,), # ensure this matches your dataset
  )

  # NOTE: Remember that EfficientNet has a built-in Normalization (Rescaling) layer

  # Create a simple Keras Sequential model with FE Layer
  model = tf.keras.Sequential([
    Rescaling(1./255), # programatically add this based on model url
    DataAugmentationLayer,
    feature_extraction_layer,
    Conv2D(10,3, activation='relu', name='custom_conv2d_layer1'),
    Conv2D(10,3, activation='relu', name='custom_conv2d_layer2'),
    MaxPool2D(name='custom_maxpool_layer1'),
    Conv2D(10,3, activation='relu', name='custom_conv2d_layer3'),
    Conv2D(10,3, activation='relu', name='custom_conv2d_layer4'),
    MaxPool2D(name='custom_maxpool_layer2'),
    Dropout(0.2, name='custom_dropout_layer1'),
    Flatten(name='custom_flat_layer1'),
    Dense(num_classes, activation='softmax', name='output_layer')
  ])

  model.build([None, 224, 224, 3]) # Batch input shape.
  # Model.build allows for model inspection before compiling and training epochs are run

  return model