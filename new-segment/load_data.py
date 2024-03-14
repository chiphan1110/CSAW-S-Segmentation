import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import *


class CSAWS(Dataset):
    def __init__(self, root_dir, label_list, single_label, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_list = label_list
        self.single_label = single_label

        self.img_dir = os.path.join(root_dir, 'img')
        self.img_paths = [os.path.join(self.img_dir, filename) for filename in sorted(os.listdir(self.img_dir))]

        self.mask_dir = os.path.join(root_dir, 'mask')
        self.mask_paths = {mask: [] for mask in self.label_list}

        for label in self.label_list:
            mask_path = os.path.join(self.mask_dir, label)
            self.mask_paths[label] = [os.path.join(mask_path, filename) for filename in sorted(os.listdir(mask_path))]
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = {label: self.mask_paths[label][idx] for label in self.label_list}

        image = Image.open(img_path)
        masks = {label: Image.open(mask_path[label]) for label in self.label_list}

        if self.transform is not None:
            image = self.transform(image)
            masks = {label: self.transform(masks[label]) for label in self.label_list}
        
        
        return image, masks[self.single_label]


if __name__ == "__main__":
    # Example usage
    # label_list = ['nipple', 'pectoral_muscle']
    label_list = ['nipple', 'pectoral_muscle']
    single_label = 'nipple'
    path = "/home/phanthc/Chi/Code/CSAW-S-Segmentation/CSAWS_Sample/CSAWS_preprocessed/test"
    transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = CSAWS(path, label_list, single_label, transform=transforms)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for data in data_loader:
        images, masks = data[0], data[1]
        print(images.shape)   #torch.Size([4, 1, 256, 256])
        print(masks.shape)    #torch.Size([4, 1, 256, 256])
        print(masks.type)