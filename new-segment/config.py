import torchvision.transforms as transforms

# Hardware Configuration 
CUDA_VISIBLE_DEVICES = "0"

# Path
DATA_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/"


# Sample path
SAMPLE_ORIGINAL_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/original_images/"
SAMPLE_MASK_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/anonymized_dataset/"
SAVE_IMG_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/img/"
SAVE_MASK_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/mask/"
SAVE_DATASET_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/"

MODEL_DIR = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/output/models/"
LOG_DIR = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/output/log/"

# Data Preprocessing and Augmentation 
IMG_SIZE = 1024  
CLAHE_FLAG = True
RESIZE_IMG = (256, 256)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Dataset Params
CLASSES = ["pectoral_muscle", "nipple"]
NB_CLASSES = len*(CLASSES)
VAL_FRACTION = 0.8

# Training Params
N_EPOCHS = 1000
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
EARLY_STOPPING = 10
WEIGHT_DECAY = 1e-4




