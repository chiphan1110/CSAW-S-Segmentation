import numpy as np
import os
import h5py
import cv2
import imutils
from tqdm import tqdm
from utils import minmaxscale
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/anonymized_dataset/"
SAVE_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/"
IMG_SIZE = 1024
CLASSES = ["_pectoral_muscle", "_nipple"]
CLAHE_FLAG = True

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def apply_clahe(img):
    """Apply CLAHE to an image."""
    clahe_create = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_create.apply(img)

def preprocess_image(file_path, img_size, clahe_flag):
    """Preprocess a single image."""
    img = cv2.imread(file_path, 0)
    if img is None:
        logging.error(f"Error reading file {file_path}. Skipping...")
        return None

    img = imutils.resize(img, height=img_size)
    orig_shape = img.shape
    
    if clahe_flag:
        img = apply_clahe(img)
    
    img = minmaxscale(img.astype(np.float32), scale_=1).astype(np.uint8)

    # Ensure image is square
    if img.shape[1] < img.shape[0]:
        tmp = np.zeros((img_size, img_size), dtype=np.uint8)
        tmp[:img.shape[0], :img.shape[1]] = img
        img = tmp

    return img, orig_shape, img.shape

def preprocess_segmentation_samples(data_path, save_path, classes, img_size, clahe_flag):
    """Preprocess segmentation samples."""
    ensure_dir(save_path)
    
    for patient in tqdm(os.listdir(data_path), desc="Processing Patients"):
        curr_path = os.path.join(data_path, patient)
        patient_save_path = os.path.join(save_path, patient)
        ensure_dir(patient_save_path)
        
        for class_ in classes:
            for file_ in os.listdir(curr_path):
                if class_ in file_:
                    file_path = os.path.join(curr_path, file_)
                    img, orig_shape, new_shape = preprocess_image(file_path, img_size, clahe_flag)
                    if img is None:
                        continue
                    
                    scan_id = file_.split(class_)[0].split("_")[0]
                    hdf5_file_path = os.path.join(patient_save_path, f"{scan_id}_{orig_shape[0]}_{orig_shape[1]}_{new_shape[0]}_{new_shape[1]}.h5")
                    with h5py.File(hdf5_file_path, "w") as f:
                        f.create_dataset(class_.strip("_"), data=img, compression="gzip", compression_opts=4)
                    logging.info(f"Processed and saved: {hdf5_file_path}")

def main():
    preprocess_segmentation_samples(DATA_PATH, SAVE_PATH, CLASSES, IMG_SIZE, CLAHE_FLAG)

if __name__ == "__main__":
    main()
