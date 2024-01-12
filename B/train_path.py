import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
from model_path import ResNet18, ResNet50
from data_preprocessing_path import load_data
import copy
import matplotlib.pyplot as plt

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.1
MODEL_SAVE_PATH = 'B/model_path.pth'


class EarlyStopping:
    """
    Implement an early stopping mechanism to prevent overfitting
    Attributes:
        patience: Number of epochs to wait after last validation loss improved.
        delta: Minimum change in the monitored quantity to qualify as an improvement.
        average_range: Number of epochs to consider for moving average of the loss.
    Methods:
        __call__(val_loss, model): Check if early stopping condition is met.
    """
    def __init__(self, patience, delta, average_range=3):
        self.patience = patience
        self.delta = delta
        self.average_range = average_range
        self.best_loss = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.losses = []

    def __call__(self, val_loss, model):
        """
        Check if early stopping is triggered and update internal state
        :param val_loss: The current validation loss
        :param model: The model being trained
        :return: True if early stopping is triggered, else False
        """
        self.losses.append(val_loss)
        if len(self.losses) > self.average_range:
            avg_loss = np.mean(self.losses[-self.average_range:])
        else:
            avg_loss = np.mean(self.losses)

        if self.best_loss is None:
            self.best_loss = avg_loss
        elif avg_loss > self.best_loss - self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = avg_loss
            self.epochs_no_improve = 0
        return self.early_stop


def train_and_validate_model(model, criterion, optimizer, scheduler, NUM_EPOCHS, train_loader, val_loader):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    early_stop = EarlyStopping(patience=5, delta=0.001)

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('=' * 15)

        # Iterate over phases: train and validate
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
                phase_desc = "Training Phase"
            else:
                model.eval()
                data_loader = val_loader
                phase_desc = "Validation Phase"

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loader, desc=f"{phase_desc} Progress"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).squeeze(1).long()
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimise in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                scheduler.step()
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if early_stop(epoch_loss, model) and phase == 'val':
            print("\nEarly stopping triggered.")
            break

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
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
    """
    Main function to execute the training process and evaluate the model,
    as well as saving the trained model to a local path.
    :return: Trained model and metrics (loss and accuracy) for training and validation phases.
    """
    print("Select the model architecture to train:")
    print("1: ResNet18")
    print("2: ResNet50")
    try:
        model_choice = int(input("Enter your choice (1 or 2): "))
        if model_choice == 1:
            model_instance = ResNet18().to(device, non_blocking=True)
        elif model_choice == 2:
            model_instance = ResNet50().to(device, non_blocking=True)
        else:
            print("Invalid choice. Defaulting to ResNet18.")
            model_instance = ResNet18().to(device, non_blocking=True)
    except ValueError:
        print("Invalid input. Defaulting to ResNet18.")
        model_instance = ResNet18().to(device, non_blocking=True)
    train_loader, val_loader, _, _ = load_data(dataset_directory='./Datasets', batch_size=BATCH_SIZE)

    # Define the loss criterion and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=LEARNING_RATE)

    # Scheduler for adjusting the learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    trained_model, train_losses, train_accs, val_losses, val_accs = train_and_validate_model(model_instance, criterion,
                                                                                             optimizer, scheduler,
                                                                                             NUM_EPOCHS, train_loader,
                                                                                             val_loader)

    # Save the best model's state to a file
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}.")

    plot_learning_curves(train_losses, train_accs, val_losses, val_accs)
    return trained_model, train_losses, train_accs, val_losses, val_accs


if __name__ == "__main__":
    model = main()
