from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch
from medmnist import PathMNIST

batch_size = 64


def count_classes(loader):
    """
    Count the number of samples in each class within the dataset
    :param loader: Dataloader containing the dataset
    :return: Tuple containing counts of normal and pneumonia samples
    """
    # Initialize a dictionary to hold count for each class
    class_counts = {
        '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0}

    for _, labels in loader:
        for label in labels:
            label = str(label.item())
            if label in class_counts:
                class_counts[label] += 1

    return class_counts


def calculate_dataset_statistics(dataset, sample_size=1000):
    """
    Calculate the mean and standard deviation across the dataset to normalise the dataset
    :param dataset: Dataset to compute statistics on
    :return: Tuple containing the mean and standard deviation of the dataset
    """
    loader = DataLoader(dataset, batch_size=sample_size, shuffle=True, num_workers=4)
    data_iter = iter(loader)
    images, _ = next(data_iter)
    mean = images.mean([0, 2, 3])
    std = images.std([0, 2, 3])
    return mean.numpy(), std.numpy()


def get_sampler_weights(dataset):
    """
    Calculate weights for each sample in the dataset to address class imbalance
    :param dataset: Dataset to calculate weights for
    :return: Tuple containing the weights for each sample
    """
    class_sample_counts = np.array([np.sum(dataset.labels == i) for i in range(len(np.unique(dataset.labels)))])
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
    :param is_train: Flag to indicate if the DataLoader is for training
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

    data = PathMNIST(root=download_directory, split=split_type, download=True, transform=transform)
    return data


def create_data_loaders(data, batch_size, is_train=True):
    """
    Create DataLoader for the given dataset
    :param data: Dataset to load
    :param batch_size: Batch size for the DataLoader
    :param is_train: Flag to indicate if the DataLoader is for training
    :return: DataLoader for the provided dataset
    """
    if is_train:
        weights = get_sampler_weights(data)
        sampler = WeightedRandomSampler(weights, len(weights))
        data_loader = DataLoader(data, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return data_loader


def data(download_directory, batch_size):
    """
    Main function to handle data loading and preprocessing
    :param download_directory: Directory to download the data
    :param batch_size: Size of the batches
    :return: Tuple containing DataLoaders and dataset statistics (mean, std).
    """
    try:
        train_dataset = PathMNIST(root=download_directory, split='train', download=True,
                                  transform=transforms.ToTensor())
        mean, std = calculate_dataset_statistics(train_dataset)

        train_dataset = load_and_preprocess_data(download_directory, 'train', mean, std, is_train=True)
        val_dataset = load_and_preprocess_data(download_directory, 'val', mean, std, is_train=False)
        test_dataset = load_and_preprocess_data(download_directory, 'test', mean, std, is_train=False)

        train_loader = create_data_loaders(train_dataset, batch_size=batch_size, is_train=True)
        val_loader = create_data_loaders(val_dataset, batch_size=batch_size, is_train=False)
        test_loader = create_data_loaders(test_dataset, batch_size=batch_size, is_train=False)

        #print(train_dataset)
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
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    try:
        train_loader, val_loader, test_loader, (mean, std) = data(download_directory, batch_size)
        class_counts = count_classes(train_loader)
        for class_id, class_name in {'0': 'adipose', '1': 'background', '2': 'debris', '3': 'lymphocytes',
                                     '4': 'mucus', '5': 'smooth muscle', '6': 'normal colon mucosa',
                                     '7': 'cancer-associated stroma',
                                     '8': 'colorectal adenocarcinoma epithelium'}.items():
            print(f"Samples in class '{class_name}' (ID: {class_id}): {class_counts[class_id]}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")