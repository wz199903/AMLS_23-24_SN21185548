from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch
from medmnist import PneumoniaMNIST

batch_size = 64


def count_classes(loader):
    """
    Count the number of samples in each class within the dataset
    :param loader: Dataloader containing the dataset
    :return: Tuple containing counts of normal and pneumonia samples
    """
    normal_count = 0
    pneumonia_count = 0

    for _, labels in loader:
        normal_count += torch.sum(labels == 0).item()
        pneumonia_count += torch.sum(labels == 1).item()

    return normal_count, pneumonia_count


def calculate_dataset_statistics(dataset):
    """
    Calculate the mean and standard deviation across the dataset to normalise the dataset
    :param dataset: Dataset to compute statistics on
    :return: Tuple containing the mean and standard deviation of the dataset
    """
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean.numpy(), std.numpy()


def get_sampler_weights(dataset):
    """
    Calculate weights for each sample in the dataset to address class imbalance
    :param dataset: Dataset to calculate weights for
    :return: Tuple containing the weights for each sample
    """
    class_sample_counts = np.unique(dataset.labels, return_counts=True)[1]
    class_weights = 1. / np.maximum(class_sample_counts, 1)
    weights = np.take(class_weights, dataset.labels)
    weights = weights.flatten()
    return torch.DoubleTensor(weights)


def load_and_preprocess_data(download_directory, split_type, mean, std, is_train=True):
    """
    Load and preprocess the data
    :param download_directory: Path to the dataset
    :param split_type: Type of the dataset split ('train', 'val', 'test')
    :param mean: mean for normalisation
    :param std: standard deviation for normalisation
    :return: Dataset with transformation
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    if is_train:
        transform_list.insert(1, transforms.RandomRotation(degrees=(-5, 5)))
        transform_list.insert(2, transforms.ColorJitter(contrast=0.1))

    transform = transforms.Compose(transform_list)

    data = PneumoniaMNIST(root=download_directory, split=split_type, download=True, transform=transform)
    return data


def create_data_loaders(data, batch_size, is_train=True):
    """
    Create dataloader for the given dataset
    :param data: Dataset to load
    :param batch_size: Batch size for the DataLoader
    :param is_train: Flag to indicate if the DataLoader is for training
    :return: Dataloader for the provided dataset
    """
    if is_train:
        weights = get_sampler_weights(data)
        sampler = WeightedRandomSampler(weights, len(weights))
        data_loader = DataLoader(data, batch_size=batch_size, sampler=sampler)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=is_train)

    return data_loader


def data(download_directory, batch_size):
    """
    Main function to handle data loading and preprocessing
    :param download_directory: Directory to download the data
    :param batch_size: Size of the batches
    :return: Tuple containing DataLoaders and dataset statistics (mean, std).
    """
    try:
        train_dataset = PneumoniaMNIST(root=download_directory, split='train', download=True,
                                       transform=transforms.ToTensor())
        mean, std = calculate_dataset_statistics(train_dataset)

        train_dataset = load_and_preprocess_data(download_directory, 'train', mean, std, is_train=True)
        val_dataset = load_and_preprocess_data(download_directory, 'val', mean, std, is_train=False)
        test_dataset = load_and_preprocess_data(download_directory, 'test', mean, std, is_train=False)

        train_loader = create_data_loaders(train_dataset, batch_size=batch_size, is_train=True)
        val_loader = create_data_loaders(val_dataset, batch_size=batch_size, is_train=False)
        test_loader = create_data_loaders(test_dataset, batch_size=batch_size, is_train=False)

        print(train_dataset)
        #print("="*50)
        #print(val_dataset)
        #print("="*50)
        #print(test_dataset)
        return train_loader, val_loader, test_loader, (mean, std)
    except Exception as e:
        print(f"An error occurred in the data function: {e}")
        raise


if __name__ == "__main__":
    download_directory = '../Datasets'
    mean, std = [0.5], [0.5]
    try:
        train_loader, val_loader, test_loader, (mean, std) = data(download_directory, batch_size)
        train_normal_count, train_pneumonia_count = count_classes(train_loader)
        print(f"Normal samples in training set: {train_normal_count}")
        print(f"Pneumonia samples in training set: {train_pneumonia_count}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")