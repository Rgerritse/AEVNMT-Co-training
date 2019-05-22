import torch
import torch.nn as nn
from .utils import rnn_creation_fn

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        rnn_fn = rnn_creation_fn(config["rnn_type"])
        self.rnn = rnn_fn(config["hidden_size"], config["hidden_size"], batch_first=True, bidirectional=True)
        self.config = config

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden) # Maybe packed sequences

        if self.config["rnn_type"] == "lstm":
            hidden = hidden[0]

        layers = [hidden[layer_num] for layer_num in range(hidden.size(0))]
        hidden_combined = torch.cat(layers, dim=-1)

        return output, hidden_combined
