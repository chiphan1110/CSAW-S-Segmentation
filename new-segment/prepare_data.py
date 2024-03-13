import numpy as np
import os
# import tensorflow as tf
# from tensorflow.keras.layers.experimental import preprocessing
# import tensorflow_addons as tfa
# from utils import *
import h5py
import cv2
import imutils
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_segmentation_samples():

    data_path = "/home/vishc1/hoang/CSAW-S/CsawS/anonymized_dataset/"
    save_path = "/home/vishc1/hoang/CSAW-S/CSAWS_preprocessed/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    classes = ["_pectoral_muscle", "_nipple"]

    img_size = 1024
    clahe = True

    for patient in tqdm(os.listdir(data_path)):
        curr_path = data_path + patient + "/"             # /home/vishc1/hoang/CSAW-S/CsawS/anonymized_dataset/099/       
        patient_save_path = save_path + patient + "/"     # /home/vishc1/hoang/CSAW-S/CSAWS_preprocessed/099/
        if not os.path.exists(patient_save_path):
            os.makedirs(patient_save_path)
        
        for class_ in classes:                              # _pectoral_muscle
            for file_ in os.listdir(curr_path):
                if class_ in file_:                         # 099_0_pectoral_muscle.png
                    scan = file_.split(class_)[0]           # 099_0
                    scan_id = scan.split("_")[0]            # 099
            
                    img = cv2.imread(curr_path + file_, 0)
                    

                
def main():
    preprocess_segmentation_samples()

if __name__ == "__main__":
    main()
