import torch
import torch.nn as nn
import torch.optim as optim
from model_pneumonia import ResNet18
from data_preprocessing_pneumonia import load_data
import matplotlib.pyplot as plt
import time
import copy
import numpy as np
from tqdm import tqdm

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'A/model_pneumonia.pth'
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.1


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


def train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS, train_loader, val_loader):
    """
    Train the model and evaluate it on the validation set
    :param model: The neural network to train
    :param criterion: The loss function to use
    :param optimizer: The optimizer to use
    :param scheduler: Learning rate scheduler
    :param NUM_EPOCHS: The number of epochs to train for
    :param train_loader: DataLoader for the training set
    :param val_loader: DataLoader for the validation set
    :return: Trained model and metrics (loss and accuracy) for training and validation phases
    """
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

            progress_bar = tqdm(data_loader, desc=f"{phase_desc} Progress")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().view_as(outputs))

                    # Backward pass and optimise in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((outputs > 0) == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                scheduler.step()

            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print(f'\n{phase_desc} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and early_stop(epoch_loss, model):
                print("\nEarly stopping triggered.")
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                model.load_state_dict(best_model_wts)
                return model, train_losses, train_accs, val_losses, val_accs

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

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
    train_loader, val_loader, _, (mean, std) = load_data(dataset_directory='./Datasets', batch_size=BATCH_SIZE)

    # Empirical weight selection
    class_weights = [0.6, 0.4]
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ResNet18().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    trained_model, train_losses, train_accs, val_losses, val_accs = train_model(model, criterion, optimizer, scheduler,
                                                                                NUM_EPOCHS, train_loader, val_loader)
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}.")

    plot_learning_curves(train_losses, train_accs, val_losses, val_accs)
    return trained_model, train_losses, train_accs, val_losses, val_accs


if __name__ == "__main__":
    model = main()
