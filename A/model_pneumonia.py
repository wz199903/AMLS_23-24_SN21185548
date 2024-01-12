import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18(nn.Module):
    """
    A custom implementation of the RenNet18 architecture for image classification
    Attributes:
        model: The ResNet18 model with custom first convolutional layer
        dropout: A dropout layer for regularisation
        fc: Redefinition of the fully connected layer for the task
    Methods:
        forward(x): Define the forward pass of the network
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        # Choose to load pretrained weights or not
        self.model = models.resnet18(weights=None)
        # self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer for 1 input channel
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.2)
        num_ftrs = self.model.fc.in_features

        # Modify the final fully connected layer for 9-class output
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 1))

    def forward(self, x):
        """
        Define the forward pass of the ResNet18
        :param x: The input tensor containing the image data
        :return: The output tensor after passing the network
        """
        x = self.model(x)

        return x

