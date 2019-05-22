import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import rnn_creation_fn

class InferenceModel(nn.Module):
    def __init__(self, config):
        super(InferenceModel, self).__init__()

        rnn_fn = rnn_creation_fn(config["rnn_type"])
        self.rnn = rnn_fn(config["hidden_size"], config["hidden_size"], batch_first=True, bidirectional=True)
        self.config = config

        self.aff_u_hid = nn.Linear(2 * config["hidden_size"], config["hidden_size"])
        self.aff_u_out = nn.Linear(config["hidden_size"], config["latent_size"])

        self.aff_s_hid = nn.Linear(2 * config["hidden_size"], config["hidden_size"])
        self.aff_s_out = nn.Linear(config["hidden_size"], config["latent_size"])

    def forward(self, x, x_mask):
        output, _ = self.rnn(x)
        avg_output = torch.sum(output * x_mask.squeeze(1).unsqueeze(-1).type_as(output), 1)
        # avg_output = torch.mean(output * x_mask.squeeze(1).unsqueeze(-1).type_as(output), 1)

        loc = self.aff_u_out(F.relu(self.aff_u_hid(avg_output)))
        scale = F.softplus(self.aff_s_out(F.relu(self.aff_s_hid(avg_output))))

        return loc, scale
