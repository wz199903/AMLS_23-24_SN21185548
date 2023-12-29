from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


import medmnist
from medmnist import PneumoniaMNIST
from medmnist import INFO, Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_flag = 'pneumoniamnist'
download = True

NUM_EPOCHS = 10
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


train_dataset = DataClass(split='train', transform=transform, download=download)
val_dataset = DataClass(split='val', transform=transform, download=download)
test_dataset = DataClass(split='test', transform=transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


class ResNet50(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        if n_channels == 1:
            # Change the first convolutional layer to accept 1-channel input
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)  # Binary classification

    def forward(self, x):
        return self.model(x)


model = ResNet50(n_channels=n_channels, n_classes=n_classes).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    model = model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float()  # Convert labels to float
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
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

model = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)

def test(model, split, task, data_loader):
    model.eval()
    y_true = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)

    data_loader = test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if task == 'binary-class':
                targets = targets.to(device=device, dtype=torch.float32)
                outputs = torch.sigmoid(outputs)
            else:
                targets = targets.squeeze().long().to(device=device)
                outputs = outputs.softmax(outputs, dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().detach().numpy()

        evaluator = medmnist.Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

        print(f'{split.capitalize()} - AUC: {metrics[0]:.3f}, Acc: {metrics[1]:.3f}')


print('==> Evaluating ...')

test(model, 'test', task, test_loader)
