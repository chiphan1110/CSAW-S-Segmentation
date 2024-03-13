import os
import numpy as np
from PIL import Image
import random
from config import *
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

    
def transform_image_and_mask(image, mask_nipple, mask_pectoral, size=(256, 256)):
    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask_nipple = TF.hflip(mask_nipple)
        mask_pectoral = TF.hflip(mask_pectoral)

    # Gamma correction (for the image only)
    if random.random() > 0.5:
        gamma = random.uniform(0.75, 1.5)
        image = TF.adjust_gamma(image, gamma, gain=1)
    
    # Resize
    image = TF.resize(image, size)
    mask_nipple = TF.resize(mask_nipple, size, interpolation=TF.InterpolationMode.NEAREST)
    mask_pectoral = TF.resize(mask_pectoral, size, interpolation=TF.InterpolationMode.NEAREST)

    # Convert to tensor
    image = TF.to_tensor(image)
    mask_nipple = TF.to_tensor(mask_nipple)
    mask_pectoral = TF.to_tensor(mask_pectoral)

    # Combine masks into a single multi-channel tensor
    mask = torch.cat((mask_nipple, mask_pectoral), dim=0)

    return image, mask


class MammogramDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
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

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('L')  
        
        mask_nipple_path = os.path.join(self.mask_dir_nipple, img_name.replace('.png', '_nipple.png'))
        mask_pectoral_muscle_path = os.path.join(self.mask_dir_pectoral_muscle, img_name.replace('.png', '_pectoral_muscle.png'))

        mask_nipple = Image.open(mask_nipple_path).convert('L')
        mask_pectoral_muscle = Image.open(mask_pectoral_muscle_path).convert('L')

        image, mask = transform_image_and_mask(img, mask_nipple, mask_pectoral_muscle)

        sample = {'image': image, 'mask': mask}

        return sample


if __name__ == "__main__":
    # Example usage
    dataset = MammogramDataset(img_dir=SAVE_IMG_PATH, mask_dir=SAVE_MASK_PATH)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for data in data_loader:
        images, masks = data['image'], data['mask']
        print(images.shape, masks.shape)  # torch.Size([batch_size, 1, H, W]) torch.Size([batch_size, 2, H, W])
