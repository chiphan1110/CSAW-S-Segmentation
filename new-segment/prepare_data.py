import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from utils import *
import h5py
import cv2
import imutils
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_segmentation_samples():

    data_path = "/home/vishc1/hoang/CSAW-S/CsawS/anonymized_dataset"
    save_path = "/home/vishc1/hoang/CSAW-S/CSAWS_preprocessed"