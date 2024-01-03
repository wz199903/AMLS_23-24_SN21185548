import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from info import INFO

batch_size = 64


class PathMNIST(Dataset):
    """
    Adapted from MedMNIST github repository
    """
    flag = "pathmnist"

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 as_rgb=False,
                 download=False):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        '''

        self.info = INFO[self.flag]
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == 'train':
            self.img = npz_file['train_images']
            self.label = npz_file['train_labels']
        elif self.split == 'val':
            self.img = npz_file['val_images']
            self.label = npz_file['val_labels']
        elif self.split == 'test':
            self.img = npz_file['test_images']
            self.label = npz_file['test_labels']

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.
        '''
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__

        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Split: {}".format(self.split))
        body.append("Task: {}".format(self.info["task"]))
        body.append("Number of channels: {}".format(self.info["n_channels"]))
        body.append("Meaning of labels: {}".format(self.info["label"]))
        body.append("Number of samples: {}".format(self.info["n_samples"]))
        body.append("Description: {}".format(self.info["description"]))
        body.append("License: {}".format(self.info["license"]))

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)


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
    Calculate the mean and standard deviation across the dataset to normalise the data set
    :param dataset: Dataset to compute statistics on
    :return: Tuple containing the mean and standard deviation of the dataset
    """
    loader = DataLoader(dataset, batch_size=sample_size, shuffle=True)
    data_iter = iter(loader)
    images, _ = next(data_iter)
    mean = images.mean([0, 2, 3])
    std = images.std([0, 2, 3])

    return mean.numpy(), std.numpy()


def get_sampler_weights(dataset, performance_metrics=None):
    """
    Calculate weights for each sample in the dataset to address class imbalance
    :param dataset: Dataset to calculate weights for
    :param performance_metrics: A dictionary with class indices as keys and performance as values
    :return: Tuple containing the weights for each sample
    """
    class_sample_counts = np.array([np.sum(dataset.label == i) for i in range(len(np.unique(dataset.label)))])
    class_weights = 1. / np.maximum(class_sample_counts, 1)
    for class_index in np.unique(dataset.label):
        if class_index in performance_metrics:
            recall = performance_metrics[class_index]['recall']
            class_weights[class_index] *= 1 / (recall + 0.1)

    weights = np.take(class_weights, dataset.label).flatten()
    return torch.DoubleTensor(weights)


def load_and_preprocess_data(dataset_directory, split_type, mean, std, is_train=True):
    """
    Load and preprocess the data
    :param dataset_directory: Path to the dataset
    :param split_type: Type of the dataset split ('train', 'val', 'test')
    :param mean: mean for normalisation
    :param std: standard deviation for normalisation
    :param is_train: Flag to indicate if the DataLoader is for training
    :return: Dataset with transformation
    """
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    if is_train:
        transform_list.insert(1, transforms.RandomRotation(degrees=(-5, 5)))
        transform_list.insert(2, transforms.ColorJitter(contrast=0.1))

    transform = transforms.Compose(transform_list)

    data = PathMNIST(root=dataset_directory, split=split_type, download=True, transform=transform)
    return data


def create_data_loaders(data, batch_size, is_train=True):
    """
    Create DataLoader for the given data set
    :param data: Dataset to load
    :param batch_size: Batch size for the DataLoader
    :param is_train: Flag to indicate if the DataLoader is for training
    :return: DataLoader for the provided dataset
    """
    performance_metrics = {
        0: {'recall': 0.97},
        1: {'recall': 1.00},
        2: {'recall': 0.77},
        3: {'recall': 0.95},
        4: {'recall': 0.81},
        5: {'recall': 0.62},
        6: {'recall': 0.77},
        7: {'recall': 0.45},
        8: {'recall': 0.92},
    }

    if is_train:
        weights = get_sampler_weights(data, performance_metrics)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        data_loader = DataLoader(data, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return data_loader


def data(dataset_directory, batch_size):
    """
    Main function to handle data loading and preprocessing
    :param dataset_directory: Directory to download the data
    :param batch_size: Size of the batches
    :return: Tuple containing DataLoaders and dataset statistics (mean, std).
    """
    try:
        train_dataset = PathMNIST(root=dataset_directory, split='train', download=True,
                                  transform=transforms.ToTensor())
        mean, std = calculate_dataset_statistics(train_dataset)
        print("Mean:", mean)
        print("Standard Deviation:", std)
        train_dataset = load_and_preprocess_data(dataset_directory, 'train', mean, std, is_train=True)
        val_dataset = load_and_preprocess_data(dataset_directory, 'val', mean, std, is_train=False)
        test_dataset = load_and_preprocess_data(dataset_directory, 'test', mean, std, is_train=False)

        train_loader = create_data_loaders(train_dataset, batch_size=batch_size, is_train=True)
        val_loader = create_data_loaders(val_dataset, batch_size=batch_size, is_train=False)
        test_loader = create_data_loaders(test_dataset, batch_size=batch_size, is_train=False)

        #print(train_dataset)
        #print("="*50)
        #print(val_dataset)
        #print("="*50)
        print(test_dataset)
        return train_loader, val_loader, test_loader, (mean, std)
    except Exception as e:
        print(f"An error occurred in the data function: {e}")
        raise


def main():
    dataset_directory = '../Datasets'
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    try:
        train_loader, val_loader, test_loader, (mean, std) = data(dataset_directory, batch_size)

        class_counts = count_classes(train_loader)
        for class_id, class_name in INFO['pathmnist']['label'].items():
            print(f"Samples in class '{class_name}' (ID: {class_id}): {class_counts[class_id]}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()