import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse
import logging
from config import *
from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--original_img_path", type=str, default=SAMPLE_ORIGINAL_PATH, help="Path to the original images")
    parser.add_argument("--original_mask_path", type=str, default=SAMPLE_MASK_PATH, help="Path to the masks")
    parser.add_argument("--save_img_path", type=str, default=SAVE_IMG_PATH, help="Path to save the preprocessed images")
    parser.add_argument("--save_mask_path", type=str, default=SAVE_MASK_PATH, help="Path to save the preprocessed masks")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Size of the preprocessed images")
    parser.add_argument("--clahe", type=bool, default=True, help="Apply CLAHE to the images")
    
    return parser.parse_args()

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

    img = img / 255.0
    
    return img

def save_data():
    args = parse_args()
    create_dir(args.save_img_path)
    create_dir(args.save_mask_path)

    for img_name in tqdm(os.listdir(args.original_img_path), desc="Processing Images"):
        if not img_name.endswith('.png'):
            continue
        base_name = img_name.split('.')[0]
        patient_id, img_id = base_name.split('_')[:2]
        img_path = os.path.join(args.original_img_path, img_name)
        
        
        original_img = preprocess_image(img_path, args.img_size, args.clahe)

        if original_img is not None:
            cv2.imwrite(os.path.join(args.save_img_path, f"{base_name}.png"), (original_img*255).astype(np.uint8))

        for class_ in CLASSES:
            mask_name = f"{patient_id}_{img_id}_{class_}.png"
            mask_path = os.path.join(args.original_mask_path, patient_id, mask_name)
            if os.path.exists(mask_path):
                mask = preprocess_image(mask_path, args.img_size)
                save_mask_class = os.path.join(args.save_mask_path, class_)
                create_dir(save_mask_class)
                if mask is not None:
                    cv2.imwrite(os.path.join(save_mask_class, f"{base_name}_{class_}.png"), (mask * 255).astype(np.uint8))
            else:
                logging.info(f"Mask not found: {mask_path}")


if __name__ == "__main__":
    save_data()
