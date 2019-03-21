import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class AEVNMT(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx, hidden_dim):
        super(AEVNMT, self).__init__()

        # Initialize parameters
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        # Initialize prior
        self.mu_prior = torch.tensor([0.0] * hidden_dim)
        self.sigma_prior = torch.tensor([1.0] * hidden_dim)
        self.normal = Normal(self.mu_prior, self.sigma_prior)

        # Initialize embeddings
        self.emb_x = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.emb_y = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.emb_x.weight = self.emb_y.weight

        # Initialize models
        self.enc = SentEmbInfModel(self.emb_x, emb_dim, hidden_dim)
        self.trans = TransModel(self.emb_x, self.emb_y, emb_dim, hidden_dim)

    def forward(self, x, y):
        mu_theta, sigma_theta = self.enc.forward(x)

        batch_size = x.shape[0]
        e = self.normal.sample(sample_shape=torch.tensor([batch_size]))
        z = mu_theta + e * sigma_theta

        self.trans.forward(x, z)
        return 0

# Currently is not conditioned on the target sentence (should be an option)
class SentEmbInfModel(nn.Module):
    def __init__(self, emb_x, emb_dim, hidden_dim):
        super(SentEmbInfModel, self).__init__()
        self.emb_x = emb_x

        # TODO: init GRU cells to zero (page 44, thesis)
        self.rnn_gru_x = nn.GRU(emb_dim, hidden_dim, batch_first = True, bidirectional=True)

        self.aff_u_hid = nn.Linear(2 * hidden_dim, hidden_dim)
        self.aff_u_out = nn.Linear(hidden_dim, hidden_dim)

        self.aff_s_hid = nn.Linear(2 * hidden_dim, hidden_dim)
        self.aff_s_out = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        f = self.emb_x(x.long())
        f.detach()
        gru_x, w = self.rnn_gru_x(f)
        h_x = torch.mean(gru_x, 1)
        mu = self.aff_u_out(self.relu(self.aff_u_hid(h_x)))
        sigma = self.softplus(self.aff_s_out(self.relu(self.aff_s_hid(h_x))))
        return mu, sigma

# Non-continious
class TransModel(nn.Module):
    def __init__(self, emb_x, emb_y, emb_dim, hidden_dim):
        super(TransModel, self).__init__()
        self.emb_x = emb_x
        self.emb_y = emb_y

        self.rnn_bigru_x = nn.GRU(emb_dim, hidden_dim, batch_first = True, bidirectional=True)
        self.aff_init_enc = nn.Linear(hidden_dim, hidden_dim)

        self.tanh = nn.Tanh()

    def forward(self, x, y, z, max_len=None):
        f = self.emb_x(x.long())
        bigru_x_h0 = torch.unsqueeze(self.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1)
        s, _ = self.rnn_bigru_x(f, bigru_x_h0)
        e = self.emb_y(y.long())
        # e_j =

        # if max_len == None:
            # max_len =

        for i in range(max_len):

    def forward_step(self):
        print("forward_step")
# class SrcLangModel(nn.Module):
#     def __init__(self, emb):
#         super().__init__()
#         self.emb = emb
#
#     def forward(self, z):
#         f_i1 = self.emb(x_i1.long())
        # if i == 0:
            # h =
        # print("forward")
:
:
