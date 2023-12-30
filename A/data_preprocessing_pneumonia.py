import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import PneumoniaMNIST

batch_size = 64


def load_and_preprocess_data(download_directory, split_type):
    try:
        data_transform = transforms.Compose([
            #transforms.RandomRotation(degrees=(-5, 5)),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        data = PneumoniaMNIST(root=download_directory, split=split_type, download=True, transform=data_transform)
        return data
    except Exception as e:
        print(f"An error occurred while loading and preprocessing the data: {e}")
        raise


#def preprocess_data(X, y):
    #X = torch.tensor(X/ 255.0, dtype=torch.float32).unsqueeze(1)
    #y = torch.tensor(y, dtype=torch.long)
    #return X, y

def create_data_loaders(data, batch_size, is_train=True):
    try:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=is_train)
        return data_loader
    except Exception as e:
        print(f"An error occurred while creating the data loader: {e}")
        raise


def data(batch_size):
    try:
        download_directory = '../Datasets'
        train_dataset = load_and_preprocess_data(download_directory, 'train')
        val_dataset = load_and_preprocess_data(download_directory, 'val')
        test_dataset = load_and_preprocess_data(download_directory, 'test')
        #pil_dataset = PneumoniaMNIST(root=download_directory, split='train', download=True)
        train_loader = create_data_loaders(train_dataset, batch_size=batch_size, is_train=True)
        val_loader = create_data_loaders(val_dataset, batch_size=batch_size, is_train=False)
        test_loader = create_data_loaders(test_dataset, batch_size=batch_size, is_train=False)

        print(train_dataset)
        print("="*50)
        print(val_dataset)
        print("="*50)
        print(test_dataset)
        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"An error occurred in the data function: {e}")
        raise


if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = data(batch_size)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")