import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from info import INFO

batch_size = 64
dataset_directory = './Datasets'


class PneumoniaMNIST(Dataset):
    """
    Adapted from MedMNIST GitHub repository and further modified for this task
    """
    flag = "pneumoniamnist"

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 ):
        """ dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
        """

        self.info = INFO[self.flag]
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

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

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):
        # Adapted from torchvision.

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
    normal_count = 0
    pneumonia_count = 0

    for _, labels in loader:
        normal_count += torch.sum(labels == 0).item()
        pneumonia_count += torch.sum(labels == 1).item()

    return normal_count, pneumonia_count


def dataset_statistics(dataset):
    """
    Calculate the mean and standard deviation across the dataset to normalise the dataset
    :param dataset: Data set to compute statistics on
    :return: Tuple containing the mean and standard deviation of the dataset
    """
    loader = DataLoader(dataset, batch_size=1)
    sum_mean = 0.
    sum_var = 0.
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        sum_mean += images.mean(2).sum(0)
        sum_var += images.var(2).sum(0)
        total_images += batch_samples

    mean = sum_mean / total_images
    std = torch.sqrt(sum_var / total_images)
    return mean.numpy(), std.numpy()


def sampler_weights(dataset, class_performance):
    """
    Calculate weights for each sample in the dataset to address class imbalance
    :param dataset: Dataset to calculate weights for
    :param class_performance: A dictionary with class indices as keys and another dictionary
                              with performance metrics as values
    :return: Tuple containing the weights for each sample
    """
    class_sample_counts = np.array([np.sum(dataset.label == i) for i in np.unique(dataset.label)])
    class_weights = 1. / np.maximum(class_sample_counts, 1)

    for class_index in class_performance:
        performance = class_performance[class_index]
        recall = performance.get('recall')
        if recall is not None:
            # Update the class weight based on recall
            class_weights[class_index] *= (1 / (recall + 0.1))
    weights = np.take(class_weights, dataset.label).flatten()
    return torch.DoubleTensor(weights)


def load_and_preprocess_data(dataset_directory, split_type, mean, std, is_train=True):
    """
    Load and preprocess the data
    :param dataset_directory: Path to the dataset
    :param split_type: Type of the dataset split ('train', 'val', 'test')
    :param mean: mean for normalisation
    :param std: standard deviation for normalisation
    :return: Dataset with transformation
    """
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    # Apply additional augmentations for training data
    if is_train:
        transform_list.insert(1, transforms.RandomRotation(degrees=(-5, 5)))
        transform_list.insert(2, transforms.ColorJitter(contrast=0.1))

    transform = transforms.Compose(transform_list)

    data = PneumoniaMNIST(root=dataset_directory, split=split_type, transform=transform)
    return data


def create_data_loaders(data, batch_size, is_train=True):
    """
    Create dataloader for the given dataset
    :param data: Dataset to load
    :param batch_size: Batch size for the DataLoader
    :param is_train: Flag to indicate if the DataLoader is for training
    :return: Dataloader for the provided dataset
    """
    class_performance = {
        0: {'recall': 0.75},
        1: {'recall': 0.99},
    }
    if is_train:
        weights = sampler_weights(data, class_performance)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        data_loader = DataLoader(data, batch_size=batch_size, sampler=sampler)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return data_loader


def load_data(dataset_directory, batch_size):
    """
    Main function to handle data loading and preprocessing
    :param dataset_directory: Directory to download the data
    :param batch_size: Size of the batches
    :return: Tuple containing DataLoaders and dataset statistics (mean, std).
    """
    try:
        train_dataset = PneumoniaMNIST(root=dataset_directory, split='train', transform=transforms.ToTensor())
        mean, std = dataset_statistics(train_dataset)
        print("Mean:", mean)
        print("Standard Deviation:", std)
        train_dataset = load_and_preprocess_data(dataset_directory, 'train', mean, std, is_train=True)
        val_dataset = load_and_preprocess_data(dataset_directory, 'val', mean, std, is_train=False)
        test_dataset = load_and_preprocess_data(dataset_directory, 'test', mean, std, is_train=False)

        train_loader = create_data_loaders(train_dataset, batch_size=batch_size, is_train=True)
        val_loader = create_data_loaders(val_dataset, batch_size=batch_size, is_train=False)
        test_loader = create_data_loaders(test_dataset, batch_size=batch_size, is_train=False)

        return train_loader, val_loader, test_loader, (mean, std)
    except Exception as e:
        print(f"An error occurred in the data function: {e}")
        raise


def main():
    try:
        train_loader, val_loader, test_loader, (mean, std) = load_data(dataset_directory, batch_size)
        train_normal_count, train_pneumonia_count = count_classes(train_loader)
        print(f"Normal samples in training set: {train_normal_count}")
        print(f"Pneumonia samples in training set: {train_pneumonia_count}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()