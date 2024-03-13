import numpy as np
import os
import cv2
from tqdm import tqdm
import logging
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def apply_clahe(img):
    clahe_create = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_create.apply(img)

def preprocess_image(img_path, img_size, clahe_flag=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.error(f"Error reading file {img_path}. Skipping...")
        return None
    
    if clahe_flag:
        img = apply_clahe(img)
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize image
    img = img / 255.0
    
    return img

def preprocess_and_save():
    ensure_dir(SAVE_IMG_PATH)
    ensure_dir(SAVE_MASK_PATH)

    for img_name in tqdm(os.listdir(ORIGINAL_IMG_PATH), desc="Processing Images"):
        if not img_name.endswith('.png'):
            continue
        base_name = img_name.split('.')[0]
        patient_id, img_id = base_name.split('_')[:2]
        img_path = os.path.join(ORIGINAL_IMG_PATH, img_name)
        
        # Preprocess and save original image
        original_img = preprocess_image(img_path, IMG_SIZE)
        if original_img is not None:
            cv2.imwrite(os.path.join(SAVE_IMG_PATH, f"{base_name}.png"), (original_img * 255).astype(np.uint8))

        # Preprocess and save masks
        for class_ in CLASSES:
            mask_name = f"{patient_id}_{img_id}_{class_}.png"
            mask_path = os.path.join(MASK_PATH, patient_id, mask_name)
            if os.path.exists(mask_path):
                mask = preprocess_image(mask_path, IMG_SIZE)
                SAVE_MASK_CLASS = os.path.join(SAVE_MASK_PATH, class_)
                ensure_dir(SAVE_MASK_CLASS)
                if mask is not None:
                    # Save individual mask
                    cv2.imwrite(os.path.join(SAVE_MASK_CLASS, f"{base_name}_{class_}.png"), (mask * 255).astype(np.uint8))
            else:
                logging.info(f"Mask not found: {mask_path}")


if __name__ == "__main__":
    preprocess_and_save()
