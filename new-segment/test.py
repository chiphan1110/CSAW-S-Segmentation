import os
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from model import UNET
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Test a MLP model')
    parser.add_argument('--init_model_file', required=True, help='Path to the trained model file', dest='init_model_file')
    parser.add_argument('--test_data_file', default='/home/vishc1/chi/Classification/data/test_data.pt', help='Path to the test data file', dest='test_data_file')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for testing')
    parser.add_argument('--output_file', default='test_predictions', help='File to save test predictions', dest='output_file')
    return parser.parse_args()

def load_test_dataset(test_data_file, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        # transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])
    
    test_data = torch.load(test_data_file)
    test_data = torch.stack([transform(img) for img in test_data])
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_model(model_file, device):
    # model = ResMLP(image_size=64, patch_size=8, num_features=512, num_classes=50)
    model = MLPMixer(image_size=64, patch_size=8, in_channels=3,num_features=128, expansion_factor=2, num_layers=8, num_classes=50, dropout=0.5)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)
    return model

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data[0].to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def save_predictions(model_file, predictions, output_file):
    output_file = f'/home/vishc1/chi/Classification/predict/{model_file.split("/")[-1].split(".")[0]}.csv'
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'label'])
        for i, prediction in enumerate(predictions, start=0):
            writer.writerow([i, prediction])

def main():
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = load_test_dataset(args.test_data_file, args.batch_size)
    model = load_model(args.init_model_file, device)

    print('Predicting on test data...')
    test_predictions = test_model(model, test_loader, device)

    save_predictions(args.init_model_file, test_predictions, args.output_file)
    print('Test evaluation completed. Predictions saved to:', args.output_file)

if __name__ == "__main__":
    main()