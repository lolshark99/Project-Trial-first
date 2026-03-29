import torch
import torch.nn as nn
class EmotionModel(nn.Module):
    def __init__(self , num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3 , 16 , 3 , padding=1)
        self.conv2 = nn.Conv2d(16 , 32 , 3 , padding=1)
        self.conv3 = nn.Conv2d(32 , 64 , 3 , padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 12 * 12, num_classes)

    def forward(self , x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x , 1)
        return self.fc(x)