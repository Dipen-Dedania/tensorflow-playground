import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# Functions and classes for loading and using the Inception model.
import Lib.inception as inception
inception.maybe_download()

model = inception.Inception()

def classify(image_path):
    # Display the image.
    # display(Image(image_path))

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10, only_first_name=True)


# image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
# classify(image_path)

classify(image_path="data/knifey-spoony/knifey/knifey-01-0015.jpg")
