import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.linear = nn.Linear(32 * 39 * 39, 50)
        self.layernorm = nn.LayerNorm(50)

    def forward(self, input):
        ninput = input / 255.0
        c1 = F.relu(self.conv1(ninput))
        c2 = F.relu(self.conv2(c1))
        flat_c2 = torch.flatten(c2)
        l1 = self.linear(flat_c2)
        n1 = self.layernorm(l1)
        out = torch.tanh(n1)
        return out