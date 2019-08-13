import torch
import torch.nn as nn
from .utils import rnn_creation_fn, tile_rnn_hidden
from torch.distributions.categorical import Categorical
import numpy as np

# class Decoder(nn.Module):
#     def __init__(self, attention, vocab_tgt_size, config):
#         super(Decoder, self).__init__()
#
#         self.attention = attention
#
#
#         rnn_fn = rnn_creation_fn(config["rnn_type"])
#         rnn_dropout = 0. if config["num_dec_layers"] == 1 else config["dropout"]
#         self.rnn = rnn_fn(config["emb_size"] + 2 * config["hidden_size"], config["hidden_size"],
#             batch_first=True, dropout=rnn_dropout, num_layers=config["num_dec_layers"])
#
#         if config["pass_enc_final"]:
#             self.init_layer = nn.Linear(2 * config["hidden_size"], config["hidden_size"])
#
#         self.dropout = nn.Dropout(config["dropout"])
#         self.pre_output_layer = nn.Linear(3 * config["hidden_size"] + config["emb_size"], config["hidden_size"])
#         self.config = config
#
#     def initialize(self, enc_output, enc_final):
#         self.attention.compute_proj_keys(enc_output)
#         if self.config["pass_enc_final"]:
#             hidden = self.init_layer(enc_final)
#             hidden = hidden.unsqueeze(0)
#             hidden = tile_rnn_hidden(hidden, self.rnn)
#             return hidden
#
#     def forward_step(self, embed_y, enc_output, x_mask, dec_hidden):
#         if self.config["rnn_type"] == "lstm":
#             query = dec_hidden[0]
#         else:
#             query = dec_hidden
#
#         query = query[-1].unsqueeze(1)
#         context, _ = self.attention.forward(query, x_mask, enc_output)
#         rnn_input = torch.cat([embed_y, context], dim=2)
#
#         dec_output, dec_hidden = self.rnn(rnn_input, dec_hidden)
#
#         pre_output = torch.cat([embed_y, dec_output, context], dim=2)
#         pre_output = self.dropout(pre_output)
#         pre_output = self.pre_output_layer(pre_output)
#         return pre_output, dec_hidden
#
#     # def forward(self, embed_y, enc_output, x_mask, dec_hidden):
#     #     outputs = []
#     #     max_len = embed_y.shape[1]
#     #     for t in range(max_len):
#     #         prev_y = embed_y[:, t:t+1, :]
#     #         pre_output, dec_hidden = self.forward_step(prev_y, enc_output, x_mask, dec_hidden)
#     #         logits = self.logits_layer(pre_output)
#     #         outputs.append(logits)
#     #     return torch.cat(outputs, dim=1)
#
#     def sample(self, emb_fn, enc_output, mask, dec_hidden, sos_idx):
#         batch_size = mask.size(0)
#         prev = mask.new_full(size=[batch_size, 1], fill_value=sos_idx,
#             dtype=torch.long)
#
#         output = [prev.squeeze(1).cpu().numpy()]
#         for t in range(self.config["max_len"]):
#              embed = emb_fn(prev)
#              pre_output, dec_hidden = self.forward_step(embed, enc_output, mask, dec_hidden)
#              # logits = self.logits_layer(pre_output) # wrong layer, use embedding matrix
#              categorical = Categorical(logits=logits)
#              next_word = categorical.sample()
#              output.append(next_word.squeeze(1).cpu().numpy())
#              prev = next_word
#              stacked_output = np.stack(output, axis=1)  # batch, time
#         return stacked_output

class Decoder(nn.Module):
    def __init__(self, attention, vocab_tgt_size, config):
        super(Decoder, self).__init__()

        self.attention = attention

        rnn_dropout = 0. if config["num_dec_layers"] == 1 else config["dropout"]
        rnn_fn = rnn_creation_fn(config["rnn_type"])
        self.rnn = rnn_fn(config["emb_size"] + 2 * config["hidden_size"], config["hidden_size"], batch_first=True, dropout=rnn_dropout, num_layers=config["num_dec_layers"])

        if config["pass_enc_final"]:
            self.init_layer = nn.Linear(2 * config["hidden_size"], config["hidden_size"])

        self.dropout = nn.Dropout(config["dropout"])
        self.pre_output_layer = nn.Linear(3 * config["hidden_size"] + config["emb_size"], config["hidden_size"])
        self.logits_layer = nn.Linear(config["hidden_size"], vocab_tgt_size, bias=False)
        self.config = config

    def initialize(self, enc_output, enc_final):
        self.attention.compute_proj_keys(enc_output)
        if self.config["pass_enc_final"]:
            hidden = self.init_layer(enc_final)
            hidden = hidden.unsqueeze(0)
            hidden = tile_rnn_hidden(hidden, self.rnn)
            return hidden

    def forward_step(self, embed_y, enc_output, x_mask, dec_hidden):
        if self.config["rnn_type"] == "lstm":
            query = dec_hidden[0]
        else:
            query = dec_hidden
        query = query[-1].unsqueeze(1)
        embed_y = embed_y.unsqueeze(1)
        context, _ = self.attention.forward(query, x_mask, enc_output)
        rnn_input = torch.cat([embed_y, context], dim=2)

        dec_output, dec_hidden = self.rnn(rnn_input, dec_hidden)

        pre_output = torch.cat([embed_y, dec_output, context], dim=2)
        pre_output = self.dropout(pre_output)
        pre_output = self.pre_output_layer(pre_output)
        return pre_output, dec_hidden

    def forward(self, embed_y, enc_output, x_mask, dec_hidden):
        outputs = []
        max_len = embed_y.shape[1]
        for t in range(max_len):
            prev_y = embed_y[:, t:t+1, :]
            pre_output, dec_hidden = self.forward_step(prev_y, enc_output, x_mask, dec_hidden)
            logits = self.logits_layer(pre_output)
            outputs.append(logits)
        return torch.cat(outputs, dim=1)
