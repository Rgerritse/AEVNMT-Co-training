import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import rnn_creation_fn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class InferenceModel(nn.Module):
    def __init__(self, config):
        super(InferenceModel, self).__init__()

        rnn_fn = rnn_creation_fn(config["rnn_type"])
        self.rnn = rnn_fn(config["hidden_size"], config["hidden_size"], batch_first=True, bidirectional=True)
        self.config = config

        if config["z_inference_mode"] == "max_hid":
            self.aff_u_hid = nn.Linear(4 * config["hidden_size"], config["hidden_size"])
            self.aff_u_out = nn.Linear(config["hidden_size"], config["latent_size"])

            self.aff_s_hid = nn.Linear(4 * config["hidden_size"], config["hidden_size"])
            self.aff_s_out = nn.Linear(config["hidden_size"], config["latent_size"])
        else:
            self.aff_u_hid = nn.Linear(2 * config["hidden_size"], config["hidden_size"])
            self.aff_u_out = nn.Linear(config["hidden_size"], config["latent_size"])

            self.aff_s_hid = nn.Linear(2 * config["hidden_size"], config["hidden_size"])
            self.aff_s_out = nn.Linear(config["hidden_size"], config["latent_size"])

    def forward(self, x, x_mask, x_len):
        packed_seq = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed_seq)
        output, _ = pad_packed_sequence(output, batch_first=True)

        if self.config["z_inference_mode"] == "max_hid":
            if self.config["rnn_type"] == "lstm":
                hidden = hidden[0]

            layers = [hidden[layer_num] for layer_num in range(hidden.size(0))]
            hidden_combined = torch.cat(layers, dim=-1)

            masked_output = output * x_mask.squeeze(1).unsqueeze(-1).type_as(output)
            max_output = torch.max(masked_output, 1)[0]
            final_output = torch.cat([max_output, hidden_combined], 1)
        else:
            final_output = torch.sum(output * x_mask.squeeze(1).unsqueeze(-1).type_as(output), 1)

        loc = self.aff_u_out(F.relu(self.aff_u_hid(final_output)))
        scale = F.softplus(self.aff_s_out(F.relu(self.aff_s_hid(final_output))))

        return loc, scale
