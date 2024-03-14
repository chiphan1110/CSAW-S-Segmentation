import os
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from load_data import MammogramDataset
from unet_torch import UNet 
from config import *



def parse_args():
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['car']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    return args


def load_dataset(args):
    full_dataset = MammogramDataset(args.data_path, args.mask_path, augmentation=get_training_augmentation(), 
)
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=args.val_fraction, random_state=42)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def training(args, train_loader, valid_loader, criterion, optimizer, model, device):
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )


    max_score = 0

    for i in range(0, 40):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

def main():
    args = parse_args()
    train_loader, val_loader = load_datasets(args)

    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    criterion = smp.utils.losses.DiceLoss()
    metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training(args, train_loader, val_loader, criterion, optimizer, model, device)


if __name__ == "__main__":
    main()