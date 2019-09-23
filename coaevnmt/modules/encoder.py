import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils import rnn_creation_fn

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        rnn_dropout = 0. if config["num_enc_layers"] == 1 else config["dropout"]
        rnn_fn = rnn_creation_fn(config["rnn_type"])
        self.rnn = rnn_fn(config["hidden_size"], config["hidden_size"], batch_first=True, bidirectional=True, dropout=rnn_dropout, num_layers=config["num_enc_layers"])
        self.config = config

    def forward(self, x, seq_len, hidden=None):
        packed_seq = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed_seq, hidden) # Maybe packed sequences
        output, _ = pad_packed_sequence(output, batch_first=True)


        if self.config["rnn_type"] == "lstm":
            hidden = hidden[0]

        layers = [hidden[layer_num] for layer_num in range(hidden.size(0))]
        hidden_combined = torch.cat(layers, dim=-1)

        return output, hidden_combined
