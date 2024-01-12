import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class ResNet18(nn.Module):
    """
    A custom implementation of the RenNet18 architecture for image classification
    Attributes:
        model: The ResNet18 model with custom first convolutional layer
        fc: Redefinition of the fully connected layer for the task
    Methods:
        forward(x): Define the forward pass of the network
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        # Choose to load pretrained weights or not
        self.model = models.resnet18(weights=None)
        # self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 3-channel images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Modify the final fully connected layer for 9-class output
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 9)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    """
    A custom implementation of the RenNet50 architecture for image classification
    Attributes:
        model: The ResNet18 model with custom first convolutional layer
        fc: Redefinition of the fully connected layer for the task
    Methods:
        forward(x): Define the forward pass of the network
    """
    def __init__(self):
        super(ResNet50, self).__init__()
        # Choose to load pretrained weights or not
        #self.model = models.resnet50(weights=None)
        #self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Modify the first convolutional layer to accept 3-channel images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Modify the final fully connected layer for 9-class output
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 9)

    def forward(self, x):
        x = self.model(x)
        return x


class SpecializedResNet50(nn.Module):
    """
    A custom implementation of the RenNet50 architecture for classifying class BACK, MUS, and STR
    Attributes:
        model: The ResNet18 model with custom first convolutional layer
        fc: Redefinition of the fully connected layer for the task
    Methods:
        forward(x): Define the forward pass of the network
    """
    def __init__(self):
        super(SpecializedResNet50, self).__init__()
        # Load a ResNet50 model
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Modify the first convolutional layer to accept 3-channel images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Modify the final fully connected layer for 3-class output (specialised classes)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        x = self.model(x)
        return x