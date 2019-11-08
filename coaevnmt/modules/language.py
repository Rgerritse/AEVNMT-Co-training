import torch
import torch.nn as nn
from .utils import rnn_creation_fn, tile_rnn_hidden

class LanguageModel(nn.Module):
    def __init__(self, vocab_src_size, config):
        super(LanguageModel, self).__init__()

        rnn_fn = rnn_creation_fn(config["rnn_type"])
        rnn_input_size = config["emb_size"]
        if config["z_feeding"]:
            rnn_input_size += config["latent_size"]
        self.rnn = rnn_fn(rnn_input_size, config["hidden_size"], batch_first=True)

        if not config["tied_embeddings"]:
            self.logits_matrix = nn.Parameter(torch.randn(vocab_src_size, config["hidden_size"]))
        self.dropout = nn.Dropout(config["dropout"])

        pre_output_input_size = config["hidden_size"] + config["emb_size"]
        if config["z_to_pre_output"]:
            pre_output_input_size += config["latent_size"]
        self.pre_output_layer = nn.Linear(pre_output_input_size, config["hidden_size"])

        self.config = config

    def forward_step(self, prev_x, hidden, z):
        rnn_input = prev_x
        if self.config["z_feeding"]:
            rnn_input = torch.cat([rnn_input, z.unsqueeze(1)], dim=2)

        output, hidden = self.rnn(rnn_input, hidden)
        pre_output = torch.cat([prev_x, output], dim=2)
        pre_output = self.dropout(pre_output)
        if self.config["z_to_pre_output"]:
            pre_output = torch.cat([pre_output, z.unsqueeze(1)], dim=2)
        pre_output = self.pre_output_layer(pre_output)
        return pre_output, hidden
