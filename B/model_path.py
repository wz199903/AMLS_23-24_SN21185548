import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # Load a pre-trained ResNet18 model
        self.model = models.resnet18(weights=None)

        # Modify the first convolutional layer to accept 3-channel images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final fully connected layer for 9-class output
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 9)

    def forward(self, x):
        """
        Define the forward pass of the ResNet18
        :param x: The input tensor containing the image data
        :return: The output tensor after passing through the network
        """
        x = self.model(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # Load a pre-trained ResNet50 model
        self.model = models.resnet50(weights=None)

        # Modify the first convolutional layer to accept 3-channel images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Modify the final fully connected layer for 9-class output
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 9)

    def forward(self, x):
        """
        Define the forward pass of the ResNet50
        :param x: The input tensor containing the image data
        :return: The output tensor after passing through the network
        """
        x = self.model(x)
        return x
