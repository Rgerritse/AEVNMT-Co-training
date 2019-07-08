import torch
import torch.nn as nn
from .utils import rnn_creation_fn, tile_rnn_hidden

class LanguageModel(nn.Module):
    def __init__(self, vocab_src_size, config):
        super(LanguageModel, self).__init__()

        rnn_fn = rnn_creation_fn(config["rnn_type"])
        self.rnn = rnn_fn(config["emb_size"], config["hidden_size"], batch_first=True)

        if not config["tied_embeddings"]:
            self.logits_matrix = nn.Parameter(torch.randn(vocab_src_size, config["hidden_size"]))
        self.dropout = nn.Dropout(config["dropout"])

    def forward_step(self, prev_x, hidden):
        output, hidden = self.rnn(prev_x, hidden)
        pre_output = self.dropout(output)
        return pre_output, hidden

    # def forward(self, embed_x, hidden):
    #     outputs = []
    #     max_len = embed_x.shape[1]
    #     for t in range(max_len):
    #         prev_x = embed_x[:, t:t+1, :]
    #         output, hidden = self.forward_step(prev_x, hidden)
    #         logits = self.logits_layer(self.dropout(output))
    #         outputs.append(logits)
    #     return torch.cat(outputs, dim=1)
