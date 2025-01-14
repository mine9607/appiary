# Appiary General Notes:

## Environment Setup

Note: This project uses a pre-defined conda env to enable GPU utilization (env_name = tensorflow)

The environment contains the following packages installed via conda-forge:

To recreate the project from the environment.yml file: `conda env create -f environment.yml`

Note: if the yml file is updated run: `conda env update -f environment.yml --prune` to update the environment


1.  Install conda (miniconda) if not yet installed: `https://docs.anaconda.com/miniconda/install/`
2.  Create a conda environment: `conda create --name <env_name>` (installs latest version of python)
    a. To install a specific python version: `conda create --name <env_name> python=3.10`
3.  List all conda environments: `conda env list`
4.  Activate the conda environment: `conda activate <env_name>`
5.  Install the necessary dependencies: `conda install conda-forge <package_name>`
6.  List all dependencies in an activated conda environment: `conda list` or `conda list -n <env_name>`
7.  List all deps installed via conda-forge: `conda list --explicit`
8.  List all deps installed via pip: `pip list`
9.  To close the conda env: `conda deactivate`

## Project Architecture:

The following features will be built into the project:

Summary Workflow:
    
**Preprocessing:**

1. Resize your images to match ResNet's input size (e.g., 224x224).
2. Normalize the pixel values as per the pre-trained model's requirements.

**Transfer Learning Workflow:**

1. Use a pre-trained ResNet model (e.g., ResNet50) without the top classification layer.
2. Freeze the base layers initially.
3. Add a custom classification head tailored to your dataset (e.g., a few dense layers and a softmax or sigmoid layer for classification).

**Training:**

4. Train the custom layers first while the base layers are frozen.
5. Optionally unfreeze some of the deeper layers and continue training with a lower learning rate.


### Data Acquisition:

    1) Images of the following apiary diseases will be obtained:
        1. Small Hive Beetles
        2. American Foulbrood
        3. European Foulbrood
        4. Wax Moth
        5. Chalkbrood
        6. Nosema
        7. Varroa Mite

    2) Images will be split into training and test directories


### Model Training:
    1) Data Pre-processing:
        A) Images will be re-sized to (224 x 224) - this must align with the Feature-Extraction layer input requirements
        B) Images will be batched into (16 imgs per batch) - optional due to small image dataset
        C) 


    2) Model Definition:
        A) Rescaling Layer (1/255.)
        B) Data Augmentation Layer
        C) Feature-extraction (Transfer Learning) Layer (ResNet50, EfficientNet, MobileNet, etc)
            - Freeze the early layers of ResNet model and use to extract features from dataset
            - Add custom layers on top to specialize in disease detection
                - remove the classification head of ResNet and replace it with fully-connected layers
            - Fine-tuning: unfreeze some of the later layers in ResNet and fine-tune on dataset.
                - allows the model to adapt pre-trained features to disease detection
        C) Conv2D Layers
        D) MaxPool, GlobalPool, Avg Pool?
        E) Repeat C & D
        F) Flatten Layer
        G) Dense Output Layer

    3) Model Compilation:

    4) Model Fit


### Model Inference:
    
1) A simple UI allowing users to upload a photo or multiple photos of hive
2) Photo will be sent to server for processing (ensure correct file type)
3) Photo will be pre-processed for Model Inference
4) Model will be passed the processed photo and make a prediction.
5) Prediction results will be sent to LLM call for guidance
6) LLM call will be sent back to client UI instructing on remedies, next steps and if local authorities need to be called

### Suggestions for Improvement:

1. Batch Size: The batch size of 16 may need adjustment based on your GPU's memory. Consider experimentation.
2. Pooling Layers: MaxPool or GlobalAveragePooling is usually sufficient; avoid overcomplicating with redundant pooling types.
3. Avoid Redundant Conv2D Blocks: Excessive convolutional layers may overfit a small dataset; focus on transfer learning and fine-tuning.
4. LLM Integration: Ensure the LLM guidance aligns with the model's predictions and provides clear, actionable advice.
5. Error Handling: Add strategies for handling uncertain predictions or low-confidence results.
