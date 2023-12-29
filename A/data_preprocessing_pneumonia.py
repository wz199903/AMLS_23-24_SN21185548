from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import medmnist
from medmnist import PneumoniaMNIST


def load_and_preprocess_data(download_directory, split_type):
    data_transform = transforms.Compose([
        #transforms.RandomRotation(degrees=(-5, 5)),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    data = PneumoniaMNIST(root=download_directory, split=split_type, download=True, transform=data_transform)
    return data


#def preprocess_data(X, y):
    #X = torch.tensor(X/ 255.0, dtype=torch.float32).unsqueeze(1)
    #y = torch.tensor(y, dtype=torch.long)
    #return X, y

def create_data_loaders(data, batch_size=64, is_train=True):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=is_train)

    return data_loader


def main():
    download_directory = '../Datasets'
    train_dataset = load_and_preprocess_data(download_directory, 'train')
    val_dataset = load_and_preprocess_data(download_directory, 'val')
    test_dataset = load_and_preprocess_data(download_directory, 'test')
    pil_dataset = PneumoniaMNIST(root=download_directory, split='train', download=True)
    train_loader = create_data_loaders(train_dataset, is_train=True)
    val_loader = create_data_loaders(val_dataset, is_train=False)
    test_loader = create_data_loaders(test_dataset, is_train=False)

    #print(train_dataset)
    #print("====================")
    #print(test_dataset)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = main()
