import math
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out

def rnn_creation_fn(rnn_type):
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return nn.GRU
    elif rnn_type == "lstm":
        return nn.LSTM

def tile_rnn_hidden(hidden, rnn):
    num_layers = rnn.num_layers
    num_layers = num_layers * 2 if rnn.bidirectional else num_layers
    hidden = hidden.repeat(num_layers, 1, 1)
    if isinstance(rnn, nn.LSTM):
        hidden = (hidden, hidden)
    return hidden


def xavier_uniform_n_(w, gain=1., n=4):
    """
    From: https://github.com/joeynmt/joeynmt/blob/master/joeynmt/initialization.py
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w: parameter
    :param gain: default 1
    :param n: default 4
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)

def init_model(model, src_pad_idx, tgt_pad_idx, config):
    print("Initializing model parameters...")
    xavier_gain = 1.

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "emb" in name:
                nn.init.normal_(param, mean=0., std=config["emb_init_std"])
                if "emb_src" in name:
                    param[src_pad_idx].zero_()
                elif "emb_tgt" in name:
                    param[tgt_pad_idx].zero_()

            elif "bias" in name:
                nn.init.zeros_(param)

            elif "rnn" in name:
                n = 4 if config["rnn_type"] == "lstm" else 3
                xavier_uniform_n_(param.data, gain=xavier_gain, n=n)

            elif len(param) > 1:
                nn.init.xavier_uniform_(param, gain=xavier_gain)
