import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import const
from PIL import Image
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=10,
                stride=10,
                padding=1,
                bias=True
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
