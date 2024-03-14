import os
import torchvision.transforms as transforms

# Hardware Configuration 
CUDA_VISIBLE_DEVICES = "0"

# Path
ROOT = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/"
DATASET_PATH = os.path.join(ROOT, "CSAWS_preprocessed/")

# Sample path
SAMPLE_TRAIN_ORIGINAL_IMG_PATH = os.path.join(ROOT, "original_images/")
SAMPLE_TRAIN_ORIGINAL_MASK_PATH = os.path.join(ROOT, "anonymized_dataset/")
SAVE_TRAIN_PATH = os.path.join(DATASET_PATH, "train/")
SAVE_TRAIN_IMG_PATH = os.path.join(SAVE_TRAIN_PATH, "img/")
SAVE_TRAIN_MASK_PATH = os.path.join(SAVE_TRAIN_PATH, "mask/")

TEST_ORIGINAL_PATH = os.path.join(ROOT, "test_data/")
TEST_ORIGINAL_IMG_PATH = os.path.join(TEST_ORIGINAL_PATH, "anonymized_dataset/")
TEST_ORIGINAL_MASK_PATH = os.path.join(TEST_ORIGINAL_PATH, "annotator_1")
SAVE_TEST_PATH = os.path.join(DATASET_PATH, "test/")
SAVE_TEST_IMG_PATH = os.path.join(SAVE_TEST_PATH, "img/")
SAVE_TEST_MASK_PATH = os.path.join(SAVE_TEST_PATH, "mask/")

OUTPUT_DIR = os.path.join(ROOT, "output/")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models/")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs/")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions/")

# Data Preprocessing and Augmentation 
IMG_SIZE = 1024  
CLAHE_FLAG = True
RESIZE_IMG = (256, 256)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Dataset Params
# CLASSES = ["pectoral_muscle", "nipple"]
CLASSES = ["nipple"]
NB_CLASSES = len(CLASSES)
VAL_FRACTION = 0.2

# Training Params
N_EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
EARLY_STOPPING = 10
WEIGHT_DECAY = 1e-4

UNET_ENCODER = 'resnet34'
UNET_ENCODER_WEIGHTS = 'imagenet'
UNET_ACTIVATION = 'softmax2d' 



