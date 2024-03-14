# Hardware Configuration 
CUDA_VISIBLE_DEVICES = "0"

# Path
DATA_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/"
SAVE_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/output/models/"
HISTORY_PATH = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/output/history/"


# Samle path
ORIGINAL_IMG_PATH = "/home/vishc1/chi/CSAW-S-Segmentation/CSAWS_Sample/original_images/"
MASK_PATH = "/home/vishc1/chi/CSAW-S-Segmentation/CSAWS_Sample/anonymized_dataset/"
SAVE_IMG_PATH = "/home/vishc1/chi/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/img/"
SAVE_MASK_PATH = "/home/vishc1/chi/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/mask/"
CLASSES = ["pectoral_muscle", "nipple"]


# Data Preprocessing and Augmentation 
IMG_SIZE = 1024  
AUG_FLAG = False  
CLAHE_FLAG = True
TRAIN_AUG = {"horz": 1, "gamma": [0.75, 1.5]}  
VAL_AUG = {}  

# Model Params 
MODEL_ARCH = "unet"  
FINE_TUNE = False  
N_EPOCHS = 1000
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
SPATIAL_DROPOUT = 0.1  
GAMMA = 3               # Focal Loss param
ACCUM_STEPS = 4  # For accumulated gradients, if used

N_PATIENTS = 150
TRAIN_VAL_SPLIT = 0.8
SHUFFLE_FLAG = True

USE_BACKGROUND = True
CLASS_NAMES = ["_pectoral_muscle", "_nipple"]
NB_CLASSES = len(CLASS_NAMES) + 1 if USE_BACKGROUND else len(CLASS_NAMES)
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1 if FINE_TUNE else 3)

