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
    parser.add_argument("--original_train_img_path", type=str, default=SAMPLE_TRAIN_ORIGINAL_IMG_PATH, help="Path to the original images")
    parser.add_argument("--original_train_mask_path", type=str, default=SAMPLE_TRAIN_ORIGINAL_MASK_PATH, help="Path to the masks")
    parser.add_argument("--processed_train_img_path", type=str, default=SAVE_TRAIN_IMG_PATH, help="Path to save the preprocessed train images")
    parser.add_argument("--processed_train_mask_path", type=str, default=SAVE_TRAIN_MASK_PATH, help="Path to save the preprocessed train masks")
    
    parser.add_argument("--original_test_img_path", type=str, default=TEST_ORIGINAL_IMG_PATH, help="Path to save the test images")
    parser.add_argument("--original_test_mask_path", type=str, default=TEST_ORIGINAL_MASK_PATH, help="Path to save the test masks")
    parser.add_argument("--processed_test_img_path", type=str, default=SAVE_TEST_IMG_PATH, help="Path to save the preprocessed train images")
    parser.add_argument("--processed_test_mask_path", type=str, default=SAVE_TEST_MASK_PATH, help="Path to save the preprocessed train masks")
    
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Size of the preprocessed images")
    parser.add_argument("--clahe", type=bool, default=CLAHE_FLAG, help="Apply CLAHE to the images")
    
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

def prepare_train_data():
    args = parse_args()
    create_dir(args.processed_train_img_path)
    create_dir(args.processed_train_mask_path)

    for img_name in tqdm(os.listdir(args.original_train_img_path), desc="Processing Train Images"):
        if not img_name.endswith('.png'):
            continue
        base_name = img_name.split('.')[0]
        patient_id, img_id = base_name.split('_')[:2]

        img_name = f"{patient_id}_{img_id}.png"
        img_path = os.path.join(args.original_train_mask_path, patient_id, img_name)
        
        
        original_img = preprocess_image(img_path, args.img_size, args.clahe)

        if original_img is not None:
            cv2.imwrite(os.path.join(args.processed_train_img_path, f"{base_name}.png"), (original_img*255).astype(np.uint8))

        for class_ in CLASSES:
            mask_name = f"{patient_id}_{img_id}_{class_}.png"
            mask_path = os.path.join(args.original_train_mask_path, patient_id, mask_name)
            if os.path.exists(mask_path):
                mask = preprocess_image(mask_path, args.img_size)
                save_mask_class = os.path.join(args.processed_train_mask_path, class_)
                create_dir(save_mask_class)
                if mask is not None:
                    cv2.imwrite(os.path.join(save_mask_class, f"{base_name}_{class_}.png"), (mask * 255).astype(np.uint8))
            else:
                logging.info(f"Mask not found: {mask_path}")

def prepare_test_data():
    args = parse_args()
    create_dir(args.processed_test_img_path)
    create_dir(args.processed_test_mask_path)

    for patient_id in tqdm(os.listdir(args.original_test_img_path), desc="Processing Test Images"):
        img_id = 0
        img_name = f"{patient_id}_{img_id}"
        img_path = os.path.join(args.original_test_img_path, patient_id, f"{img_name}.png")

        original_test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
        if original_test_img is not None:
            cv2.imwrite(os.path.join(args.processed_test_img_path, f"{img_name}.png"), (original_test_img*255).astype(np.uint8))

        for class_ in CLASSES:
            mask_name = f"{patient_id}_{img_id}_{class_}.png"
            mask_path = os.path.join(args.original_test_mask_path, patient_id, mask_name)
            if os.path.exists(mask_path):
                mask = preprocess_image(mask_path, args.img_size)
                save_test_mask_class = os.path.join(args.processed_test_mask_path, class_)
                create_dir(save_test_mask_class)
                if mask is not None:
                    cv2.imwrite(os.path.join(save_test_mask_class, f"{img_name}_{class_}.png"), (mask * 255).astype(np.uint8))
            else:
                logging.info(f"Mask not found: {mask_path}")

if __name__ == "__main__":
    prepare_train_data()
    prepare_test_data()
