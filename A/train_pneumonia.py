import torch
import torch.nn as nn
import torch.optim as optim
from model_pneumonia import ResNet50
from data_preprocessing_pneumonia import data, load_and_preprocess_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

NUM_EPOCHS = 50
BATCH_SIZE = 64
lr = 0.001
MODEL_SAVE_PATH = './model_pneumonia.pth'
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.1
K_FOLDS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS, train_loader, val_loader):
    model = model.to(device)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('_' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
                print("Training Phase")
            else:
                model.eval()
                data_loader = val_loader
                print("Validation Phase")

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.view_as(outputs))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((outputs > 0) == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model, train_losses, train_accs, val_losses, val_accs

def plot_learning_curves(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 5))

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


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main():
    train_loader, val_loader, _ = data(batch_size=BATCH_SIZE)
    model = ResNet50().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    trained_model, train_losses, train_accs, val_losses, val_accs = train_model(model, criterion, optimizer,scheduler, NUM_EPOCHS, train_loader,val_loader)
    save_model(trained_model, MODEL_SAVE_PATH)
    plot_learning_curves(train_losses, train_accs, val_losses, val_accs)

    return trained_model, train_losses, train_accs, val_losses, val_accs


if __name__ == "__main__":
    model = main()
