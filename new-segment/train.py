import os
import datetime
import torch
import torch.nn as nn
import torch.optim 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
from load_data import CSAWS
from model import UNet 
from config import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train model for image segmentation")
    parser.add_argument("--dataset_dir", type=str, default=SAVE_DATASET_PATH, help="Path to the dataset directory")
    parser.add_argument("--label_list", type=list, default=CLASSES, help="List of labels")
    parser.add_argument("--transform", type=transforms.Compose, default=TRANSFORM, help="Data augmentation")
    parser.add_argument("--val_fraction", type=float, default=VAL_FRACTION, help="Fraction of the dataset to use for validation")
    
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--nclasses", type=int, default=NB_CLASSES, help="Number of classes")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of epochs")
    parser.add_argument("--early_stopping", type=int, default=EARLY_STOPPING, help="Early stopping")
    
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Path to the model directory")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Path to the log directory")
    
    args = parser.parse_args()

    return args


def initialize_training_env(args):
    create_dir(args.model_dir)
    create_dir(args.log_dir)
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M%_S")
    log_file = os.path.join(args.log_dir, f"log_{current_time}.txt")
    return log_file


def load_dataset(args):
    full_dataset = CSAWS(args.dataset_dir, args.label_list, args.transform)
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=args.val_fraction, random_state=42)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    training_loss = 0.0

    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item() * images.size(0)
    
    training_loss = training_loss / len(train_loader.dataset)
    return training_loss


def validate(model, valid_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(valid_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item() * images.size(0)
    
    val_loss = val_loss / len(valid_loader.dataset)
    return val_loss


def training(args, train_loader, valid_loader, criterion, optimizer, model, device):
    early_stopping = args.early_stopping
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M%_S")
    log_file = initialize_training_env(args)
    initialize_metrics_file(log_file)

    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        training_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, valid_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improvement = 0
        else:
            improvement += 1
        
        if improvement > early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        log_metrics(log_file, epoch+1, training_loss, val_loss)
    
    final_model_file = os.path.join(args.model_dir, f"model_{current_time}.pth")
    save_model(model, args.epoch, final_model_file)
    print("Training complete!")


def main():
    args = parse_args()
    train_loader, val_loader = load_dataset(args)

    model = UNet(n_channels=3, n_classes=args.nclasses).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training(args, train_loader, val_loader, criterion, optimizer, model, device)


if __name__ == "__main__":
    main()