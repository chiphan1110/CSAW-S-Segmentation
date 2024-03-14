import os
import torch


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_metrics_file(metrics_file):
    with open(metrics_file, 'w') as f:
        f.write("epoch\ttraining_loss\tval_loss\n")

def log_metrics(metrics_file, epoch, training_loss, val_loss):
    with open(metrics_file, 'a') as f:
        f.write(f"{epoch}\t{training_loss:.4f}\t{val_loss:.4f}\n")


def save_model(model, model_file):
    state_dict = model.state_dict()
    torch.save(state_dict, model_file)
