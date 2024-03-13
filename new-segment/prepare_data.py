import numpy as np
import os
import h5py
import cv2
import imutils
from tqdm import tqdm
from utils import minmaxscale
import matplotlib.pyplot as plt

def preprocess_segmentation_samples():

    data_path = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/anonymized_dataset/"
    save_path = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    classes = ["_pectoral_muscle", "_nipple"]

    img_size = 1024
    clahe = True

    for patient in tqdm(os.listdir(data_path)):
        curr_path = data_path + patient + "/"             # /home/vishc1/hoang/CSAW-S/CsawS/anonymized_dataset/005/       
        patient_save_path = save_path + patient + "/"     # /home/vishc1/hoang/CSAW-S/CSAWS_preprocessed/005/
        if not os.path.exists(patient_save_path):
            os.makedirs(patient_save_path)
        
        for class_ in classes:                              # _pectoral_muscle
            for file_ in os.listdir(curr_path):
                if class_ in file_:                         # 005_0_pectoral_muscle.png
                    scan = file_.split(class_)[0]           # 005_0
                    scan_id = scan.split("_")[0]            # 005
            
                    img = cv2.imread(curr_path + file_, 0)
                    orig_shape = img.shape                  # (3584, 2816)
                    img = imutils.resize(img, height=img_size)
                    new_shape = img.shape                   # (1024, 1024)                       

                    # HDF5 file, store processed images along with labels, original shape, and new shape
                    f = h5py.File(patient_save_path + scan_id + "_" + str(orig_shape[0]) + "_" + str(orig_shape[1]) +
                                  "_" + str(new_shape[0]) + "_" + str(new_shape[1]) + ".h5", "w")

                    if clahe:                               # Apply Contrast Limited Adaptive Histogram Equalization
                        clahe_create = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        img = clahe_create.apply(img)
                    
                    img = minmaxscale(img.astype(np.float32), scale_=1).astype(np.uint8)    #normalize img to [0, 1] 
                    
                    if img.shape[1] < img.shape[0]:
                        tmp = np.zeros((img_size, img_size), dtype=np.uint8)
                        img_shapes = img.shape
                        tmp[:img_shapes[0], :img_shapes[1]] = img
                        img = tmp                           # (1024, 1024)

                    f.create_dataset(class_.strip("_"), data=img, compression="gzip", compression_opts=4)
                    f.close()
                
def main():
    preprocess_segmentation_samples()

if __name__ == "__main__":
    main()
