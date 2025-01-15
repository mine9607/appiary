from fastapi import APIRouter, File, UploadFile, HTTPException
import tensorflow as tf
import os

IMAGE_SIZE=(224,224)

router = APIRouter()

def load_and_prep_image(image_bytes, img_shape=IMAGE_SIZE):
  '''
  Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, color_channels)
  '''

  # Read in the raw image (returns a tensor with the entire contents of the input file)
  # img_bytes = tf.io.read_file(filename)

  # Decode the read file into a tensor (converts input bytes string into a tensor)
  img = tf.image.decode_image(image_bytes,channels=3) # tensor of type 'dtype'

  print("Initial shape: ", img.shape)
  # Resize the image
  img = tf.image.resize(img, size=img_shape)
  print("Final shape: ", img.shape)
  # Rescale the image
  img = img/255.

  print(f"The processed image: {img}")

  return img





@router.post("/process-image")
async def process_image(img_bytes):
  """
  Receive an image and preps it for use in model prediction
  """
  processed_image = load_and_prep_image(img_bytes)
  return processed_image