# Appiary General Notes:

## Environment Setup

Note: This project uses a pre-defined conda env to enable GPU utilization (env_name = tensorflow)

The environment contains the following packages installed via conda-forge:

To recreate the project from the environment.yml file: `conda env create -f environment.yml`

Note: if the yml file is updated run: `conda env update -f environment.yml --prune` to update the environment

1. Install conda (miniconda) if not yet installed: `https://docs.anaconda.com/miniconda/install/`
2. Create a conda environment:
   `conda create --name <env_name>` (installs latest version of python)
   `conda create --name <env_name> python=3.10` (install a specific python version)
3. List all conda environments: `conda env list`
4. Activate the conda environment: `conda activate <env_name>`
5. Install the necessary dependencies: `conda install conda-forge <package_name>`
6. List all dependencies in an activated conda environment: `conda list` or `conda list -n <env_name>`
7. List all deps installed via conda-forge: `conda list --explicit`
8. List all deps installed via pip: `pip list`
9. To close the conda env: `conda deactivate`

---

## Project Architecture:

The following features will be built into the project:

**PRE-PROCESSING:**

1. Resize your images to match ResNet's input size (e.g., 224x224).
2. Normalize the pixel values as per the pre-trained model's requirements.

**TRANSFER LEARNING WORKFLOW:**

1. Use a pre-trained ResNet model (e.g., ResNet50) without the top classification layer.
2. Freeze the base layers initially.
3. Add a custom classification head tailored to your dataset (e.g., a few dense layers and a softmax or sigmoid layer for classification).

**TRAINING:**

4. Train the custom layers first while the base layers are frozen.
5. Optionally unfreeze some of the deeper layers and continue training with a lower learning rate.

---

### Data Acquisition:

1. Images of the following apiary diseases will be obtained:

   - Small Hive Beetles
   - American Foulbrood
   - European Foulbrood
   - Wax Moth
   - Chalkbrood
   - Varroa Mite

2. Images will be split into training and test directories

---

### Model Training:

1. **Data Pre-processing:**

   - Images will be re-sized to (224 x 224) - this must align with the Feature-Extraction layer input requirements
   - Images will be batched into (16 imgs per batch) - optional due to small image dataset

2. **Model Definition:**

> _Data Augmentation Layer:_

```python
Rescaling(1./255, input_shape = (224, 224, 3))
RandomFlip(mode='horizontal_and_vertical')
RandomRotation(0.2)
RandomZoom(0.2)
RandomTranslation(0.2, 0.2)
```

> _Feature-extraction (Transfer Learning) Layer (ResNet50, EfficientNet, MobileNet, etc)_

    1. Freeze the early layers of ResNet model and use to extract features from dataset
    2. Add custom layers on top to specialize in disease detection
    3. remove the classification head of ResNet and replace it with fully-connected layers
    4. Fine-tuning: unfreeze some of the later layers in ResNet and fine-tune on dataset.
       - allows the model to adapt pre-trained features to disease detection

> _Conv2D Layers (10, 3, activation='relu') -> (filters, kernel size, padding, strides, activation)_

1. Filters: - Decides how many filters should pass over an input tensor -> (10, 32, 64, 128)

   - Higher values lead to more complex models

2. Filter Size: - Determines the shape of the filters (sliding window) -> (Typical values: 3, 5, 7)

   - Lower values learn smaller features -> higher values learn larger features

3. Padding:

   - 'same' = pads the target tensor with zeroes
   - 'valid' = leaves the target tensor as is (lowering output shape).

4. Strides: - The number of steps a filter takes across an image at a time

   - Example (strides = 1 : a filter moves across an image 1 pixel at a time)

> _MaxPool, GlobalPool, Avg Pool_

> _Repeat Step 3 & 4_

> _Dropout Layer (0.2) -> prevent overfitting_

> _Flatten Layer_

> _Dense Output Layer (10, activation='softmax', name='outputs')_

3. **Model Compilation:**

```python
Optimizer = "adam"
Loss Model = "tf.keras.losses.CategoricalCrossentropy()"
Metrics = ['accuracy']
```

4. **Model Fit:**
   - train_data
   - batch_size = 16 (optional)
   - epochs = optimization parameter (start with 5)
   - steps_per_epoch = len(train_data)
   - validation_batch_size = 16 (optional)
   - validation_data = test_data
   - validation_steps = len(test_data)

### Model Inference:

1. A simple UI allowing users to upload a photo or multiple photos of hive
2. Photo will be sent to server for processing (ensure correct file type)
3. Photo will be pre-processed for Model Inference
4. Model will be passed the processed photo and make a prediction.
5. Prediction results will be sent to LLM call for guidance
6. LLM call will be sent back to client UI instructing on remedies, next steps and if local authorities need to be called

### Suggestions for Improvement:

1. Batch Size: The batch size of 16 may need adjustment based on your GPU's memory. Consider experimentation.
2. Pooling Layers: MaxPool or GlobalAveragePooling is usually sufficient; avoid overcomplicating with redundant pooling types.
3. Avoid Redundant Conv2D Blocks: Excessive convolutional layers may overfit a small dataset; focus on transfer learning and fine-tuning.
4. LLM Integration: Ensure the LLM guidance aligns with the model's predictions and provides clear, actionable advice.
5. Error Handling: Add strategies for handling uncertain predictions or low-confidence results.

### Feature Extraction:

Using Tensorflow Hub (now Kaggle)

EfficientNet URL:

- `https://www.kaggle.com/models/google/efficientnet-v2/frameworks/TensorFlow2/variations/imagenet1k-b0-feature-vector/versions/2`

ResNet50 URL:

- `https://www.kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-feature-vector/versions/2`

**Incorporating Feature Extraction:**

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

def create_model(model_url, num_classes=10, image_size=(224, 224)):
  '''
  Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in the output layer,
    image_size (tuple): The height and width of images input to the model

  Returns:
    An uncompiled Keras Sequential model with model_url as feature extractor
    layer and Dense output layer with num_classes output neurons.
  '''

  # Ensure the feature extraction layer uses the correct image size

  # Turn the feature extraction model vector into a Keras Layer
  feature_extraction_layer = hub.KerasLayer(model_url,
                                            trainable=False,
                                            name='feature_extraction_layer',
                                            input_shape=image_size + (3,)) # Ensure this matches your dataset

  # Create a simple Keras Sequential model with FE Layer
  model = tf.keras.Sequential([
      Rescaling(1./255),
      feature_extraction_layer,
      tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
  ])

  model.build([None, 224, 224, 3])  # Batch input shape.

  return model

# Compile the Resnet model
resnet_model.compile('adam','categorical_crossentropy', ['accuracy'])

# Fit the Resnet model
resnet_history = resnet_model.fit(train_data,
                                  epochs=5,
                                  steps_per_epoch=len(train_data),
                                  validation_data=test_data,
                                  validation_steps=len(test_data),
                                  callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub',
                                                                         experiment_name='resnet50v2')]
                                  )

```
