import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
