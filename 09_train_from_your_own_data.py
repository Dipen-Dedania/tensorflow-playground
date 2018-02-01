import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

# Functions and classes for loading and using the Inception model.
import inception

# We use Pretty Tensor to define the new classifier.
import prettytensor as pt

import Lib.knifey as knifey
from Lib.knifey import num_classes

knifey.maybe_download_and_extract()

dataset = knifey.load()

class_names = dataset.class_names
print(class_names)
