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

    return args


def load_dataset(args):
    full_dataset = MammogramDataset(args.data_path, args.mask_path)
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
        
        # do something (save model, change lr, etc.)
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
    criterion = smp.utils.losses.DiceLoss()

    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training(args, train_loader, val_loader, criterion, optimizer, model, device)

























    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_datasets(DATA_PATH, MASK_PATH, TRAIN_VAL_SPLIT)

    model = UNet(n_channels=3, n_classes=NB_CLASSES).to(device)  

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()  # Adjust based on your loss function
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    num_epochs = N_EPOCHS  # Set the number of epochs

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Turn off gradients for validation
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        # Print training and validation loss
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "unet_model.pth"))
if __name__ == "__main__":
    main()