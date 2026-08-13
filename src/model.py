import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 3 * 64 * 64

        # block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)       # 3 * 64 * 64 -> 32 * 64 * 64
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)      # 32 * 64 * 64 -> 32 * 64 * 64
        self.bn2 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(2, 2)                   # 32 * 64 * 64 -> 32 * 32 * 32

        # block 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)      # 32 * 32 * 32 -> 64 * 32 * 32
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)      # 64 * 32 * 32 -> 64 * 32 * 32
        self.bn4 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(2, 2)                   # 64 * 32 * 32 -> 64 * 16 * 16

        # block 3
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)     # 64 * 16 * 16 -> 128 * 16 * 16
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)    # 128 * 16 * 16 -> 128 * 16 * 16
        self.bn6 = nn.BatchNorm2d(128)

        self.pool3 = nn.MaxPool2d(2, 2)                   # 128 * 16 * 16 -> 128 * 8 * 8

        # classifier
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 200)

    def forward(self, X):
        # block 1
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.relu(self.bn2(self.conv2(X)))
        X = self.pool1(X)

        # block 2
        X = F.relu(self.bn3(self.conv3(X)))
        X = F.relu(self.bn4(self.conv4(X)))
        X = self.pool2(X)

        # block 3
        X = F.relu(self.bn5(self.conv5(X)))
        X = F.relu(self.bn6(self.conv6(X)))
        X = self.pool3(X)

        # flatten
        X = torch.flatten(X, 1)

        # fully connected
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.fc2(X)

        return X