import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0)):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
