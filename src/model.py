import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 * 64 * 64

        # block 1
        # 3 * 64 * 64 -> 32 * 64 * 64
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 32 * 64 * 64 -> 32 * 64 * 64
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 32 * 64 * 64 -> 32 * 32 * 32
        self.pool1 = nn.MaxPool2d(2, 2)

        # block 2
        # 32 * 32 * 32 -> 64 * 32 * 32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 64 * 32 * 32 -> 64 * 32 * 32
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # 64 * 32 * 32 -> 64 * 16 * 16
        self.pool2 = nn.MaxPool2d(2, 2)

        # block 3
        # 64 * 16 * 16 -> 128 * 16 * 16
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        # 128 * 16 * 16 -> 128 * 16 * 16
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # 128 * 16 * 16 -> 128 * 8 * 8
        self.pool3 = nn.MaxPool2d(2, 2)

        # block 4
        # 128 * 8 * 8 -> 256 * 8 * 8
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        # 256 * 8 * 8 -> 256 * 8 * 8
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        # 256 * 8 * 8 -> 256 * 4 * 4
        self.pool4 = nn.MaxPool2d(2, 2)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))       # 256 * 4 * 4 -> 256 * 1 * 1
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 200)

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

        # block 4
        X = F.relu(self.bn7(self.conv7(X)))
        X = F.relu(self.bn8(self.conv8(X)))
        X = self.pool4(X)

        # classifier
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.dropout(X)
        X = self.fc1(X)

        return X
