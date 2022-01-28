import torch.nn as nn
from conv_block import Block


class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            Block(in_channels, 32, (3, 3), (2, 2)),
            Block(32, 128, (5, 5), (2, 2)),
            Block(128, 128, (3, 3), (2, 2)),
            nn.Dropout2d(0.2),
            Block(128, 64, (1, 1), (2, 2)),
            nn.Dropout2d(p=0.2),
            Block(64, 32, (1, 1), (1, 1)),
            nn.Flatten(),
            nn.Linear(10368, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.model(x)
