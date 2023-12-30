import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # Considering the images are 28x28 and grayscale, input channels = 1.
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After three pooling layers, the image size will be reduced to 28 / 2 / 2 / 2 = 3.5, which rounds down to 3 pixels.
        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Applying convolutions, activation functions, and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flattening the output for the fully connected layers
        x = x.view(-1, 64 * 3 * 3)  # Flatten the tensor

        # Fully connected layers with a ReLU activation function and a dropout layer for some regularization
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Using sigmoid for the binary classification task

        return x




#class ResNet50(nn.Module):
    #def __init__(self):
        #super(ResNet50, self).__init__()
        #self.model = models.resnet18(weights=None)
        #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.dropout = nn.Dropout(p=0.5)
        #num_ftrs = self.model.fc.in_features
        #self.model.fc = nn.Sequential(
            #nn.Linear(num_ftrs, 1),
        #)

    #def forward(self, x):
        #x = self.model(x)
        #return x

