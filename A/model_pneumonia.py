import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class CNN(nn.Module):
    """
    A custom CNN for image classification task
    Attributes:
        conv1-5: Convolutional layers with varying numbers of channels
        pool: A MaxPool2d layer for downsampling the feature maps
        fc1, fc2: Fully connected layers for classification

    Methods:
        forward(x): Defines the forward pass of the network
    """
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=256 * 3 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        """
        Defines the forward pass of the CNN
        :param x: The input tensor containing the image data
        :return: The output tensor after passing through the network
        """
        # Applying convolutions, activation functions, and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # Flattening the output for the fully connected layers
        x = x.view(-1, 256 * 3 * 3)  # Flatten the tensor

        # Fully connected layers with a ReLU activation function and a dropout layer for some regularization
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ResNet18(nn.Module):
    """
    A custom implementation of the RenNet18 architecture for image classification
    Attributes:
        model: The base ResNet18 model with custom first convolutional layer and batch normalisation
        dropout: A dropout layer for regularisation
        fc: Redefinition of the fully connected layer for the task

    Methods:
        forward(x): Define the forward pass of the network
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.relu = nn.ReLU(inplace=True)
        self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
        )

    def forward(self, x):
        """
        Define the forward pass of the ResNet18
        :param x: The input tensor containing the image data
        :return: The output tensor after passing the network
        """
        x = self.model(x)
        return x

