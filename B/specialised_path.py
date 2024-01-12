import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from model_path import SpecializedResNet50
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time
from torchvision import transforms
from data_preprocessing_path import PathMNIST

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

dataset_directory = './Datasets'
MODEL_SAVE_PATH = 'B/specialized_model_path.pth'


def filter_dataset_for_classes(dataset, classes):
    indices = []

    for i, (_, label) in enumerate(dataset):
        # Check if the label is in the target classes
        if label in classes:
            indices.append(i)

    return Subset(dataset, indices)


def count_filtered_classes(loader):
    """
    Count the number of samples in each of the specified classes within the dataset
    :param loader: DataLoader containing the filtered dataset
    :return: Dictionary with counts for classes 2, 5, and 7 (now 0, 1, 2)
    """
    class_counts = {'0': 0, '1': 0, '2': 0}

    for _, labels in loader:
        for label in labels:
            label_str = str(label.item())
            if label_str in class_counts:
                class_counts[label_str] += 1

    return class_counts


def remap_labels(labels, class_mapping):
    remapped_labels = torch.tensor([class_mapping[label.item()] for label in labels])
    return remapped_labels


def load_and_preprocess_data(dataset_directory, split_type, is_train=True):
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7393936, 0.5312033, 0.70391774], std=[0.12475868, 0.17564684, 0.12466431])
    ]
    if is_train:
        augmentations = [
            transforms.RandomRotation((-5, 5)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        ]
        transform_list = transform_list[:1] + augmentations + transform_list[1:]

    transform = transforms.Compose(transform_list)

    data = PathMNIST(root=dataset_directory, split=split_type, transform=transform)
    return data


def train_and_validate_model(model, criterion, optimizer, scheduler, NUM_EPOCHS, train_loader, val_loader):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('=' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loader, desc=f"{phase} Progress"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if labels.dim() > 1:
                    labels = labels.squeeze(1)
                labels = labels.long()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_losses, train_accs, val_losses, val_accs


def plot_learning_curves(train_losses, train_accs, val_losses, val_accs):
    """
    Plot the learning curves of training and validation phases
    :param train_losses: List of training losses
    :param train_accs: List of training accuracies
    :param val_losses: List of validation losses
    :param val_accs: List of validation accuracies
    """
    plt.figure(figsize=(12, 5))

    train_accs = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_accs]
    val_accs = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in val_accs]

    # Plot training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    train_dataset = load_and_preprocess_data(dataset_directory, 'train', is_train=True)
    val_dataset = load_and_preprocess_data(dataset_directory, 'val', is_train=False)

    class_mapping = {2: 0, 5: 1, 7: 2}

    # Filter datasets for classes 2, 5, and 7
    train_dataset_filtered = filter_dataset_for_classes(train_dataset, [2, 5, 7])
    val_dataset_filtered = filter_dataset_for_classes(val_dataset, [2, 5, 7])

    # Then create DataLoaders for these filtered datasets
    train_loader = DataLoader(train_dataset_filtered, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda batch: (torch.stack([item[0] for item in batch]),
                                                        remap_labels(
                                                            torch.stack([torch.tensor(item[1]) for item in batch]),
                                                            class_mapping)))

    val_loader = DataLoader(val_dataset_filtered, batch_size=BATCH_SIZE,
                            collate_fn=lambda batch: (torch.stack([item[0] for item in batch]),
                                                      remap_labels(
                                                          torch.stack([torch.tensor(item[1]) for item in batch]),
                                                          class_mapping)))

    train_class_counts = count_filtered_classes(train_loader)
    val_class_counts = count_filtered_classes(val_loader)
    print("Training set class counts:", train_class_counts)
    print("Validation set class counts:", val_class_counts)

    # Define the specialized model
    model = SpecializedResNet50().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train and validate the model
    model, train_losses, train_accs, val_losses, val_accs = train_and_validate_model(model, criterion, optimizer,
                                                                                     scheduler, NUM_EPOCHS,
                                                                                     train_loader, val_loader)

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Plot learning curves
    plot_learning_curves(train_losses, train_accs, val_losses, val_accs)


if __name__ == "__main__":
    main()
