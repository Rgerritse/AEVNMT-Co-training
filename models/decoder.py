import torch
import torch.nn as nn
from .utils import rnn_creation_fn, make_hidden_state

class Decoder(nn.Module):
    def __init__(self, attention, config):
        super(Decoder, self).__init__()

        self.attention = attention

        rnn_fn = rnn_creation_fn(config["rnn_type"])
        self.rnn = rnn_fn(config["emb_size"] + 2 * config["hidden_size"], config["hidden_size"], batch_first=True)

        if config["pass_enc_final"]:
            self.init_layer = nn.Linear(2 * config["hidden_size"], config["hidden_size"])

        self.dropout = nn.Dropout(config["dropout"])
        self.pre_output_layer = nn.Linear(3 * config["hidden_size"] + config["emb_size"], config["hidden_size"])
        self.config = config

    def initialize(self, enc_output, enc_final):
        self.attention.compute_proj_keys(enc_output)
        if self.config["pass_enc_final"]:
            hidden = self.init_layer(enc_final)
            hidden = make_hidden_state(hidden, self.config["rnn_type"])
            return hidden

    def forward_step(self, embed_y, enc_output, x_mask, dec_hidden):
        if self.config["rnn_type"] == "lstm":
            query = dec_hidden[0].unsqueeze(1)
            h_n = dec_hidden[0].unsqueeze(0)
            c_n = dec_hidden[1].unsqueeze(0)
            dec_hidden = (h_n, c_n)
        else:
            query = dec_hidden.unsqueeze(1)
            dec_hidden = dec_hidden.unsqueeze(0)

        context, _ = self.attention.forward(query, x_mask, enc_output)
        rnn_input = torch.cat([embed_y, context], dim=2)

        dec_output, dec_hidden = self.rnn(rnn_input, dec_hidden)

        if self.config["rnn_type"] == "lstm":
            h_n = dec_hidden[0].squeeze(0)
            c_n = dec_hidden[1].squeeze(0)
            dec_hidden = (h_n, c_n)
        else:
            dec_hidden = dec_hidden.squeeze(0)

        pre_output = torch.cat([embed_y, dec_output, context], dim=2)
        pre_output = self.dropout(pre_output)
        pre_output = self.pre_output_layer(pre_output)
        return pre_output, dec_hidden
