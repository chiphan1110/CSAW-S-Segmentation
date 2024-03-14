import os
import torch
import matplotlib.pyplot as plt


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_train_log_file(metrics_file):
    with open(metrics_file, 'w') as f:
        f.write("epoch\ttrain_dice_loss\tval_dice_loss\ttrain_iou\tval_iou\n")

def initialize_test_log_file(metrics_file):
    with open(metrics_file, 'w') as f:
        f.write("test_dice_loss\ttest_iou\n")

def save_model(model, model_file):
    torch.save(model, model_file)

def visualize(image, ground_truth_mask, predicted_mask):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
