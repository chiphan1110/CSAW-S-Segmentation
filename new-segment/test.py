import os
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
from load_data import CSAWS
from config import *
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Test a UNET model')
    parser.add_argument('--init_model_file', required=True, help='Path to the trained model file', dest='init_model_file')
    parser.add_argument('--test_data_dir', default=SAVE_TEST_PATH, help='Path to the test data file', dest='test_data_file')
    parser.add_argument('--label_list', type=list, default=CLASSES, help='List of labels', dest='label_list')
    
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='input batch size for testing')
    parser.add_argument('--transform', type=transforms.Compose, default=TRANSFORM, help='Data augmentation', dest='transform')
    parser.add_argument('--output_file', default=PRED_DIR, help='File to save test predictions', dest='output_file')
    
    return parser.parse_args()

def load_test_dataset(args):
    
    test_dataset = CSAWS(args.test_data_dir, args.label_list, args.transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return test_loader

def load_model(model_file, device):
    # model = UNET(in_channels=3, classes=NB_CLASSES)
    # state_dict = torch.load(model_file, map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])
    # model.to(device)

    model = torch.load(model_file)
    return model

def test_model(best_model, test_loader, loss, metrics, device):
    # model.eval()
    # predictions = []
    # with torch.no_grad():
    #     for data in test_loader:
    #         data = data[0].to(device)
    #         outputs = model(data)
    #         _, predicted = torch.max(outputs, 1)
    #         predictions.extend(predicted.cpu().numpy())
    # return predictions

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=device,
    )
    logs = test_epoch.run(test_loader)
    print(logs)

    # for img in test_loader:
    #     img = img.to(device)
    #     pred = best_model(img)
    #     pred = pred.cpu().detach().numpy()




    
# def save_predictions(model_file, predictions, output_file):
#     output_file = 
#     with open(output_file, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['id', 'label'])
#         for i, prediction in enumerate(predictions, start=0):
#             writer.writerow([i, prediction])

def main():
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = load_test_dataset(args.test_data_dir, args.batch_size)
    criterion = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    model = load_model(args.init_model_file, device)

    print('Predicting on test data...')
    test_model(model, test_loader, criterion, metrics, device)


if __name__ == "__main__":
    main()