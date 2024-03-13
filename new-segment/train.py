import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow_addons as tfa
from datetime import datetime
from models import get_arch, Unet
# from batch_generator import get_dataset, batch_gen
# from utils import macro_accuracy, flatten_
from tensorflow.keras.optimizers import Adam
# from resunetpp import *

# paths
data_path =  "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/"  
save_path = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/new-segment/output/models/"
history_path = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/new-segment/output/history/"