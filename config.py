from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
STATIC_DIR_PATH = PROJECT_ROOT / "src" / "static"
TEMPLATE_DIR_PATH = PROJECT_ROOT / "src" / "templates"

MODEL_DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" # save directory for ML models
RESNET_MODEL_URL = "https://www.kaggle.com/models/google/resnet-v2/TensorFlow2/50-feature-vector/2"
IMAGE_SIZE = (224, 224)

PORT = 8000
