import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

class AEVNMT(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx, hidden_dim, device, train=False, s_tensor=None):
        super(AEVNMT, self).__init__()

        self.device = device

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
        self.attention = BahdanauAttention(hidden_dim, query_size = 2 * hidden_dim + emb_dim)
        self.source = SourceModel(self.emb_x, emb_dim, hidden_dim, vocab_size, train=train, s_tensor=s_tensor).to(device)
        self.trans = TransModel(self.emb_x, self.emb_y, emb_dim, hidden_dim, vocab_size, self.attention, train=train, s_tensor=s_tensor).to(device)
        self.enc = SentEmbInfModel(self.emb_x, emb_dim, hidden_dim).to(device)

    def forward(self, x, x_mask, y=None, y_mask=None):
        mu_theta, sigma_theta = self.enc.forward(x)

        batch_size = x.shape[0]
        e = self.normal.sample(sample_shape=torch.tensor([batch_size])).to(self.device)
        z = mu_theta + e * sigma_theta

        pre_out_x = self.source.forward(z, x=x)
        pre_out_y = self.trans.forward(x, x_mask, z, y=y)

        return pre_out_x, pre_out_y, mu_theta, sigma_theta

class SourceModel(nn.Module):
    def __init__(self, emb_x, emb_dim, hidden_dim, vocab_size, train=False, s_tensor=None):
        super(SourceModel, self).__init__()
        self.s_tensor = s_tensor
        self.train = train
        self.emb_x = emb_x

        # self.tanh = nn.Tanh()
        self.aff_init_lm = nn.Linear(hidden_dim, hidden_dim)
        self.rnn_gru_lm = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.aff_out_x = nn.Linear(hidden_dim, vocab_size)
    def forward(self, z, x=None, max_len=None):
        if max_len == None:
            max_len = x.shape[-1]

        pre_out = []

        h = [torch.tanh(self.aff_init_lm(z))]
        for j in range(max_len):
            if self.train == True:
                f_j = self.emb_x(torch.unsqueeze(x[:, j], 1).long())
            else:
                raise NotImplementedError
            h_j, _ = self.rnn_gru_lm(f_j, h[j].unsqueeze(0))
            h.append(torch.squeeze(h_j))
            pre_out.append(self.aff_out_x(h[j]))
        return pre_out

# Non-continious
class TransModel(nn.Module):
    def __init__(self, emb_x, emb_y, emb_dim, hidden_dim, vocab_size, attention, train=False, s_tensor=None):
        super(TransModel, self).__init__()
        self.s_tensor = s_tensor
        self.emb_x = emb_x
        self.emb_y = emb_y
        self.t_size = hidden_dim * 2 + emb_dim

        self.aff_init_enc = nn.Linear(hidden_dim, hidden_dim)
        self.rnn_bigru_x = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.attention = attention

        self.aff_init_dec = nn.Linear(hidden_dim, self.t_size)
        self.rnn_gru_dec = nn.GRU(self.t_size, self.t_size, batch_first=True)

        self.aff_out_y = nn.Linear(self.t_size + emb_dim, vocab_size)
        self.train = train

    def forward(self, x, x_mask, z, y=None, max_len=None):
        f = self.emb_x(x.long())
        bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
        s, _ = self.rnn_bigru_x(f, bigru_x_h0)

        if max_len == None:
            max_len = y.shape[-1]

        t = [torch.tanh(self.aff_init_dec(z))]
        proj_key = self.attention.key_layer(s)

        pre_out = []
        for j in range(max_len):
            c_j, _ = self.attention(t[j].unsqueeze(1), proj_key, s, x_mask)
            if self.train == True:
                e_j = self.emb_y(torch.unsqueeze(y[:, j], 1).long())
            else:
                raise NotImplementedError

            h0 = torch.cat((c_j, e_j), 2)
            t_j, _ = self.rnn_gru_dec(t[j].unsqueeze(1), torch.squeeze(h0).unsqueeze(0))
            t.append(torch.squeeze(t_j))
            pre_out.append(self.aff_out_y(torch.cat((t_j, e_j), 2)))
        return pre_out

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

    def forward(self, x):
        f = self.emb_x(x.long())
        f.detach()
        gru_x, w = self.rnn_gru_x(f)
        h_x = torch.mean(gru_x, 1)
        mu = self.aff_u_out(F.relu(self.aff_u_hid(h_x)))
        sigma = F.softplus(self.aff_s_out(F.relu(self.aff_s_hid(h_x))))
        return mu, sigma

# Not tested yet
class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas
