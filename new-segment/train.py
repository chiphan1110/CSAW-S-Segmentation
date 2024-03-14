import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import argparse
from tqdm import tqdm
from load_data import CSAWS
from config import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train model for image segmentation")
    parser.add_argument("--dataset_dir", type=str, default=SAVE_TRAIN_PATH, help="Path to the dataset directory")
    parser.add_argument("--label_list", type=list, default=CLASSES, help="List of labels")
    parser.add_argument("--single_label", type=str, default=SINGLE_LABEL, help="Single label")
    parser.add_argument("--transform", type=transforms.Compose, default=TRANSFORM, help="Data augmentation")
    parser.add_argument("--val_fraction", type=float, default=VAL_FRACTION, help="Fraction of the dataset to use for validation")
    
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--nclasses", type=int, default=NB_CLASSES, help="Number of classes")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of epochs")
    parser.add_argument("--early_stopping", type=int, default=EARLY_STOPPING, help="Early stopping")
    
    parser.add_argument("--encoder", type=str, default=UNET_ENCODER, help="Encoder")
    parser.add_argument("--encoder_weights", type=str, default=UNET_ENCODER_WEIGHTS, help="Encoder weights")
    parser.add_argument("--activation", type=str, default=UNET_ACTIVATION, help="Activation function")

    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Path to the model directory")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Path to the log directory")
    
    args = parser.parse_args()

    return args


def initialize_training_env(args):
    create_dir(args.model_dir)
    create_dir(args.log_dir)
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_file = os.path.join(args.log_dir, f"log_{current_time}.txt")
    return log_file


def load_dataset(args):
    full_dataset = CSAWS(args.dataset_dir, args.label_list, args.single_label, args.transform)
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=args.val_fraction, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def training(args, train_loader, valid_loader, criterion, optimizer, metrics, model, device):
    early_stopping = args.early_stopping
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M%_S")
    log_file = initialize_training_env(args)
    initialize_train_log_file(log_file)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    max_score = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement > early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1,}\t{train_logs['dice_loss']:.6f}\t{train_logs['iou_score']:.6f}\t{valid_logs['dice_loss']:.6f}\t{valid_logs['iou_score']:.6f}\n")
    
    final_model_file = os.path.join(args.model_dir, f"model_{args.single_label}_{current_time}.pth")
    save_model(model, final_model_file)
    print("Training complete!")

def main():
    args = parse_args()
    train_loader, val_loader = load_dataset(args)

    model = smp.Unet(
        encoder_name=args.encoder, 
        encoder_weights=args.encoder_weights, 
        in_channels=1,
        classes=len(args.label_list), 
        activation=args.activation,
    )

    criterion = utils.losses.DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics = [utils.metrics.IoU(threshold=0.5),]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training(args, train_loader, val_loader, criterion, optimizer, metrics, model, device)


if __name__ == "__main__":
    main()