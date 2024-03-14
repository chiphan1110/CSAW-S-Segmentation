import os
import numpy as np
from PIL import Image
import random
from config import *
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as albu


# def get_training_augmentation():
#     train_transform = [
#         # Simple horizontal flips might be beneficial since mammogram positioning can vary,
#         # and there's no strict lateral orientation in some cases.
#         albu.HorizontalFlip(p=0.5),
        
#         # Mild rotations and shifts can help the model generalize to slight variations in positioning.
#         # Use conservative limits to avoid losing important features.
#         albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.05, p=0.8, border_mode=0),
        
#         # Ensure the image is a minimum size, might be necessary depending on your dataset.
#         albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        
#         # Random cropping can simulate closer views of the region of interest.
#         # Be cautious with the size to avoid losing important context.
#         albu.RandomCrop(height=320, width=320, always_apply=False, p=0.5),

#         # Mild elastic transformations can simulate the natural variability in breast tissue appearance.
#         albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        
#         # Random brightness and contrast adjustments can help the model adapt to different imaging conditions.
#         albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),

#         # Normalize the images (ensure the mean and std are appropriate for your dataset)
#         albu.Normalize(mean=(0.5,), std=(0.5,)),
#     ]
#     return albu.Compose(train_transform)

# def get_validation_augmentation():
#     """Adjust size to make the image shape divisible by 32 for easier downscaling in the model."""
#     test_transform = [
#         albu.Resize(384, 384, always_apply=True),
#     ]
#     return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    # Ensure there's a channel dimension, expected shape [H, W, C]
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=-1)
    # Move channel dimension to the front, convert to float32
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class MammogramDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augmentation=None, preprocessing=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.mask_dir_nipple = os.path.join(mask_dir, 'nipple')
        self.mask_dir_pectoral_muscle = os.path.join(mask_dir, 'pectoral_muscle')
        self.img_names = os.listdir(img_dir)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert('L'))  
        
        mask_nipple_path = os.path.join(self.mask_dir_nipple, img_name.replace('.png', '_nipple.png'))
        mask_pectoral_muscle_path = os.path.join(self.mask_dir_pectoral_muscle, img_name.replace('.png', '_pectoral_muscle.png'))

        mask_nipple = np.array(Image.open(mask_nipple_path).convert('L'))
        mask_pectoral_muscle = np.array(Image.open(mask_pectoral_muscle_path).convert('L'))

        mask = np.stack([mask_nipple, mask_pectoral_muscle], axis=0)  
        
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
        if self.preprocessing:
            processed = self.preprocessing(image=image, mask=mask)
            image, mask = processed['image'], processed['mask']
        
        return {'image': image, 'mask': mask}

def preprocessing_fn_normalize(img):
    """Normalize image data to [0, 1] range."""
    return img / 255.0


if __name__ == "__main__":
    # Example usage
    
    dataset = MammogramDataset(img_dir=SAVE_IMG_PATH, mask_dir=SAVE_MASK_PATH,  preprocessing=get_preprocessing(lambda x: preprocessing_fn_normalize(x)))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for data in data_loader:
        images, masks = data['image'], data['mask']
        print(images.shape, masks.shape)  # torch.Size([batch_size, 1, H, W]) torch.Size([batch_size, 2, H, W])
