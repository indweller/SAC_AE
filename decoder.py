import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(50, 32 * 39 * 39)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=3, kernel_size=3, stride=2,output_padding=1)
    
    def forward(self, input):
        l1 = F.relu(self.linear(input))
        l2 = torch.reshape(l1, [32, 39, 39])
        d1 = F.relu(self.deconv1(l2))
        out = self.deconv2(d1)
        return out