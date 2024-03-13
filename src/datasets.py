import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from copy import deepcopy
import random

from .helpfuns import *
from albumentations import *
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader


def compute_stats(dataloader):
    from tqdm import tqdm
    channels = dataloader.dataset[0]['img'].size(0)
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    for x in tqdm(dataloader):
        x_tot += x['img'].mean([0,2,3]).cpu().numpy()
        x2_tot += (x['img']**2).mean([0,2,3]).cpu().numpy()

    channel_avr = x_tot/len(dataloader)
    channel_std = np.sqrt(x2_tot/len(dataloader) - channel_avr**2)
    return channel_avr,channel_std


def order_dict(unordered_dict, decreasing=False):
    return {k: v for k, v in sorted(unordered_dict.items(), 
                                    key=lambda item: item[1], reverse=decreasing)}

def get_ordered_names(unordered_dict, int_to_labels):
    ordered_dict = order_dict(unordered_dict)
    return [int_to_labels[key] for key in ordered_dict.keys()]    
    
def get_ordered_ids(unordered_dict):    
    ordered_dict = order_dict(unordered_dict)
    return list(ordered_dict.keys())

def get_mask_pixel_count(dataloader, main_target = 'person'):
    from tqdm import tqdm
    main_target_id = dataloader.dataset.labels_to_int[main_target]
    images_target_and_label = {key:0 for key in dataloader.dataset.int_to_labels.keys()}
    imgs_with_class = {key:0 for key in dataloader.dataset.int_to_labels.keys()}
    total_N_pixels = 0
    total_pixels = {key:[] for key in dataloader.dataset.int_to_labels.keys()}
    per_image_pixels = {key:[] for key in dataloader.dataset.int_to_labels.keys()}
    for x in tqdm(dataloader):
        mask = x['mask']
        ids, counts = mask.unique(return_counts=True)
        ids = ids.cpu().tolist()
        counts = counts.cpu().numpy()
        total_N_pixels += mask.view(-1).size(0)
        for idx, countx in zip(ids, counts):
            total_pixels[idx].append(countx)
        for m in mask:
            ids, counts = m.unique(return_counts=True)
            ids = ids.cpu().tolist()
            counts = counts.cpu().numpy()
            counts = counts / m.view(-1).size(0)
            for idx, countx in zip(ids, counts):
                per_image_pixels[idx].append(countx)
                imgs_with_class[idx] += 1
                if  main_target_id in ids:
                    images_target_and_label[idx] += 1               

    avg_pixels = {}
    for key, val in total_pixels.items():
        avg_pixels[key] = np.array(val).sum() / total_N_pixels
    avg_per_image_pixels = {}
    mask_presence = {}
    for key, val in images_target_and_label.items():
        images_target_and_label[key] /= imgs_with_class[main_target_id]    
    for key, val in per_image_pixels.items():
        mask_presence[key] = len(val) / len(dataloader.dataset)
        avg_per_image_pixels[key] = np.array(val).mean()
        imgs_with_class[key] /= len(dataloader.dataset)

    return avg_pixels, avg_per_image_pixels, mask_presence, imgs_with_class, images_target_and_label

def print_by_order_and_name(dict_stat, int_to_labels, decreasing=True):
    ordered_dict_stat = order_dict(dict_stat, decreasing)
    for key, val in ordered_dict_stat.items():
        print(int_to_labels[key], round(val*100,3))

def pil_loader(img_path):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def cv2_loader(img_path):
    img = cv.imread(img_path)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB) 

    
class BaseSet(Dataset):
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])

    def get_trans_list_album(self, transform_dict):
        transform_list = []            
            
        if transform_dict['RandomRotate90']:
            transform_list.append(RandomRotate90(p=0.5))
            
        if transform_dict['Resize']['apply']:
            transform_list.append(Resize(height=transform_dict['Resize']['height'],
                                         width=transform_dict['Resize']['width'],
                                         interpolation=cv.INTER_NEAREST))
            
        if transform_dict['CenterCrop']['apply']:
            transform_list.append(CenterCrop(height=transform_dict['CenterCrop']['height'],
                                         width=transform_dict['CenterCrop']['width']))
            
        if transform_dict['RandomCrop']['apply']:
            transform_list.append(RandomCrop(height=transform_dict['RandomCrop']['height'],
                                         width=transform_dict['RandomCrop']['width'])) 
            
        if transform_dict['RandomBrightnessContrast']['apply']:
            temp_d = transform_dict['RandomBrightnessContrast']
            transform_list.append(RandomBrightnessContrast(brightness_limit=temp_d['brightness_limit'], 
                                                           contrast_limit=temp_d['contrast_limit']))
            
        if transform_dict['RandomGamma']['apply']:
            transform_list.append(RandomGamma(gamma_limit=transform_dict['RandomGamma']['gamma_limit']))  
            
        if transform_dict['Flip']:
            transform_list.append(Flip(p=1.))
            
        if transform_dict['RandomRotatons']['apply']:
            transform_list.append(Rotate(limit=transform_dict['RandomRotatons']['angle']))  
            
        if transform_dict['ElasticTransform']['apply']:
            alpha = transform_dict['ElasticTransform']['alpha']
            sigma = alpha * transform_dict['ElasticTransform']['sigma']
            alpha_affine = alpha * transform_dict['ElasticTransform']['alpha_affine']
            transform_list.append(ElasticTransform(p=0.5, alpha=alpha, 
                                                   sigma=sigma, alpha_affine=alpha_affine))           
            
        if transform_dict['Normalize']['apply']:
            transform_list.append(Normalize(mean=transform_dict['Normalize']['mean'], 
                                            std=transform_dict['Normalize']['std']))   
        
        return transform_list

    def get_transfomrs(self):
        
        if self.mode == 'train':
            aplied_transforms = self.train_transforms
        if self.mode == 'eval':
            aplied_transforms = self.val_transforms
        if self.mode == 'test':
            aplied_transforms = self.test_transforms
    
        transformations = self.get_trans_list_album(aplied_transforms)
        transforms = Compose(transformations)
        return transforms
    
    def overide_json_stats(self):
        self.train_transforms['Normalize']['mean'] = self.mean
        self.train_transforms['Normalize']['std'] = self.std
        self.val_transforms['Normalize']['mean'] = self.mean
        self.val_transforms['Normalize']['std'] = self.std
        self.test_transforms['Normalize']['mean'] = self.mean
        self.test_transforms['Normalize']['std'] = self.std
        
        
class CsawSet(BaseSet):
    def __init__(self, dataset_params, mode='train', seed_n=None, included_labels='all'):
        self.attr_from_dict(dataset_params)
        self.annotator_id = 'annotator_{}'.format(self.annotator_id)
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.root_dir = os.path.join(self.root_dir, 'patches', 'crop_size_{}'.format(self.crop_size))
        self.split_file_path = os.path.join(self.data_location, self.dataset_location, 
                                            'training_random_splits.json')
        self.bootstrap_splits = load_json(self.split_file_path)        
        self.init_stats()
        self.mode = mode
        self.data = self.get_dataset()
        self.transforms = self.get_transfomrs()
        self.tt = ToTensor()

        self.just_zero_others = False
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        img_path = self.data[idx]['img_path']
        mask_path = self.data[idx]['mask_path']
        label = self.data[idx]['label']
        
        if self.mode == 'test' and self.test_on_gold_standard:
            def_path = os.path.join(*mask_path.split('/')[:-3])
            def_path = os.path.join('/', def_path, 'binary_masks')
            pid = mask_path.split('/')[-1]            
            mask_1 = self.get_x(os.path.join(def_path, 'annotator_1', pid))
            mask_2 = self.get_x(os.path.join(def_path, 'annotator_2', pid))
            mask_3 = self.get_x(os.path.join(def_path, 'annotator_3', pid))
            
            mask = np.zeros_like(mask_1)
            mask_sum = mask_1 + mask_2 + mask_3
            mask[mask_sum > 1] = 1
            
            if not self.is_binary:
                mask[mask == 0] = self.labels_to_int['background']
                mask[mask == 1] = self.labels_to_int['cancer']
        else:
            mask = self.get_x(mask_path)
            
        img = self.get_x(img_path)
        augments = self.transforms(image=img, mask=mask)
        
        img = self.tt(augments['image'])
        mask = self.tt(augments['mask']).squeeze(0).mul(256).long()
            
        return {'img': img, 'mask': mask, 
                'label': idx}
    
    def get_x(self, img_path):
        return cv.imread(img_path, cv.IMREAD_ANYDEPTH).astype('uint8')
    
    def get_data_as_list(self, data_loc):
        data_list = []
        fnames = [f for f in os.listdir(os.path.join(data_loc, 'images')) if f.endswith('.png')]
        if not self.use_full_training_set and self.mode == 'train':
            try:
                subset = self.bootstrap_splits[str(self.how_many_samples)][str(self.subset_n)]
            except:
                raise IOError(
                    "This bootstrap split is not defined.\nPlease create a bootstrap split for {} samples."
                    .format(self.how_many_samples))
                
            fnames = [name for name in fnames if name.split("-")[0]  in subset]
            
        for fname in fnames:
            img_path = os.path.join(data_loc, 'images', fname)
            if self.is_binary:
                mask_path = os.path.join(data_loc, 'binary_masks')
            else:
                mask_path = os.path.join(data_loc, 'masks')  
            if self.mode == 'test':
                mask_path = os.path.join(mask_path, self.annotator_id)   
            mask_path = os.path.join(mask_path, fname)
            label = int(fname.split('-')[-2]) if self.mode == 'train' else 0
            data_list.append({'img_path':img_path, 'mask_path': mask_path, 'label':label})
            
        return data_list
    
    def get_dataset(self):
        if self.mode == 'train':
            self.d_path = os.path.join(self.root_dir, 'train')
        elif self.mode == 'eval':
            self.d_path = os.path.join(self.root_dir, 'val')                
        elif self.mode == 'test':
            self.d_path = os.path.join(self.root_dir, 'test')               
        return self.get_data_as_list(self.d_path)    
    
    def init_stats(self):
        self.mean = [0.176]
        self.std = [0.218]           
        self.overide_json_stats()
        
    def binarize_labels(self):
            self.int_to_labels = {0: 'background',
                                  1: 'cancer'} 
            
    img_channels = 1
    int_to_labels = {0: 'cancer',
                     1: 'calcifications',
                     2: 'axillary_lymph_nodes',
                     3: 'thick_vessels',
                     4: 'foreign_object',
                     5: 'skin',
                     6: 'nipple',
                     7: 'text',
                     8: 'non-mammary_tissue',
                     9: 'pectoral_muscle',
                     10: 'mammary_gland',
                     11: 'background'}  
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
