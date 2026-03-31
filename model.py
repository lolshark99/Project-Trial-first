import torch
import torch.nn as nn

class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)

        # Block 2
        self.conv2_1 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(32)

        # Block 3
        self.conv3_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 12 * 12, 96)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = torch.relu(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = torch.relu(x)

        x = self.pool(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = torch.relu(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = torch.relu(x)

        x = self.pool(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = torch.relu(x)

        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = torch.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc2(x)