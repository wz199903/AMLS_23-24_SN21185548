import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_V2_S_Weights


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # Load a ResNet18 model
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


class EfficientNet_V2(nn.Module):
    def __init__(self):
        super(EfficientNet_V2, self).__init__()
        # Load a pre-trained EfficientNet model
        self.model = models.efficientnet_v2_s(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 9)


    def forward(self, x):
        """
        Define the forward pass of the EfficientNetV2
        :param x: The input tensor containing the image data
        :return: The output tensor after passing through the network
        """
        x = self.model(x)
        return x


class PathCNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units * 4,
                      kernel_size=3),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 4,
                      out_channels=hidden_units * 4,
                      kernel_size=3),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 4,
                      out_channels=hidden_units * 4,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))

        self.fc = nn.Sequential(
            nn.Linear(hidden_units * 4 * 4 * 4, hidden_units * 8),
            nn.ReLU(),
            nn.Linear(hidden_units * 8, hidden_units * 8),
            nn.ReLU(),
            nn.Linear(hidden_units * 8, 9))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x