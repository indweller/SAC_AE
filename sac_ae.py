import torch.nn as nn
import torch.nn.functional as F
import encoder

def weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data)
        layer.bias.data.fill_(0.0)
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        layer.weight.data.fill_(0.0)
        layer.bias.data.fill_(0.0)
        mid = layer.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(layer.weight.data[:, :, mid, mid], gain)

class Actor(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=1024, action_dim=4):
        super().__init__()

        self.encoder = encoder.Encoder()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 2 * action_dim)

        self.apply(weight_init)

    def forward(self, input):
        e1 = self.encoder(input)
        l1 = F.relu(self.linear1(e1))
        l2 = F.relu(self.linear2(l1))
        mu, std = self.linear3(l2).chunk(2, dim=-1)

        return mu, std
