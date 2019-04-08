import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import copy

class AEVNMT(nn.Module):
    def __init__(self, vocab, vocab_size, emb_dim, padding_idx, hidden_dim, max_len, device, train=False, sos_idx=None, eos_idx=None):
        super(AEVNMT, self).__init__()

        self.device = device
        self.vocab = vocab
        # Initialize parameters
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.max_len = max_len

        # Initialize priors
        self.mu_prior = torch.tensor([0.0] * hidden_dim)
        self.sigma_prior = torch.tensor([1.0] * hidden_dim)
        self.normal = Normal(self.mu_prior, self.sigma_prior)

        # Initialize embeddings
        self.emb_x = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.emb_y = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.emb_x.weight = self.emb_y.weight

        # Initialize models
        self.attention = BahdanauAttention(hidden_dim, query_size = 2 * hidden_dim + emb_dim)
        self.source = SourceModel(self.emb_x, emb_dim, hidden_dim, vocab_size, train=train, sos_idx=sos_idx, eos_idx=eos_idx).to(device)
        self.trans = TransModel(self.vocab, self.emb_x, self.emb_y, emb_dim, hidden_dim, vocab_size, self.attention, device, train=train, sos_idx=sos_idx, eos_idx=eos_idx).to(device)
        self.enc = SentEmbInfModel(self.emb_x, emb_dim, hidden_dim).to(device)

    def forward(self, x, x_mask, y=None, y_mask=None):
        mu_theta, sigma_theta = self.enc.forward(x)

        batch_size = x.shape[0]
        e = self.normal.sample(sample_shape=torch.tensor([batch_size])).to(self.device)
        z = mu_theta + e * sigma_theta

        pre_out_x = self.source.forward(z, x=x)
        pre_out_y = self.trans.forward(x, x_mask, z, y=y)

        return pre_out_x, pre_out_y, mu_theta, sigma_theta

    def predict(self, x, x_mask):
        with torch.no_grad():
            mu_theta, sigma_theta = self.enc.forward(x)
            predictions = self.trans.greedy_decode(x, x_mask, mu_theta, self.max_len)
            return predictions
            # print(mu_theta.shape)

class SourceModel(nn.Module):
    def __init__(self, emb_x, emb_dim, hidden_dim, vocab_size, train=False, sos_idx=None, eos_idx=None):
        super(SourceModel, self).__init__()
        self.sos_idx = sos_idx
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
    def __init__(self, vocab, emb_x, emb_y, emb_dim, hidden_dim, vocab_size, attention, device, train=False, sos_idx=None, eos_idx=None):
        super(TransModel, self).__init__()
        self.vocab = vocab
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
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
        self.device = device

    def forward(self, x, x_mask, z, y=None, max_len=None):
        print("-- Train")
        # print("y: ", y)
        batch_size = x.shape[0]

        f = self.emb_x(x.long())
        bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
        s, _ = self.rnn_bigru_x(f, bigru_x_h0)

        if max_len == None:
            max_len = y.shape[-1]

        t = [torch.tanh(self.aff_init_dec(z))]
        proj_key = self.attention.key_layer(s)

        pre_out = []
        for j in range(max_len):
            # print("Timestep: ", j)
            c_j, _ = self.attention(t[j].unsqueeze(1), proj_key, s, x_mask)
            if j == 0:
                start_seq = torch.tensor([[self.sos_idx] for _ in range(batch_size)]).to(self.device)
                # print("input: \n", self.vocab.string(start_seq))
                e_j = self.emb_y(start_seq.long())
            else:
                # print("input: \n", self.vocab.string(torch.unsqueeze(y[:, j-1], 1)))
                e_j = self.emb_y(torch.unsqueeze(y[:, j], 1).long())

            h0 = torch.cat((c_j, e_j), 2)
            t_j, _ = self.rnn_gru_dec(t[j].unsqueeze(1), torch.squeeze(h0).unsqueeze(0))
            t.append(torch.squeeze(t_j))
            pre_out.append(self.aff_out_y(torch.cat((t_j, e_j), 2)))
        return pre_out

    def greedy_decode(self, x, x_mask, z, max_len):
        # print("-- Decode")
        batch_size = x.shape[0]

        f = self.emb_x(x.long())
        bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
        s, _ = self.rnn_bigru_x(f, bigru_x_h0)

        if max_len == None:
            max_len = y.shape[-1]

        t = [torch.tanh(self.aff_init_dec(z))]
        proj_key = self.attention.key_layer(s)

        sequences = torch.tensor([[self.sos_idx] for _ in range(batch_size)]).to(self.device)
        for j in range(max_len):
            # print("Timestep: ", j)
            # print("Input: \n", self.vocab.string(torch.unsqueeze(sequences[:, j], 1)))
            c_j, _ = self.attention(t[j].unsqueeze(1), proj_key, s, x_mask)
            e_j = self.emb_y(torch.unsqueeze(sequences[:, j], 1).long())

            h0 = torch.cat((c_j, e_j), 2)
            t_j, _ = self.rnn_gru_dec(t[j].unsqueeze(1), torch.squeeze(h0).unsqueeze(0))
            t.append(torch.squeeze(t_j))
            pre_out_j = self.aff_out_y(torch.cat((t_j, e_j), 2))
            probs_j = F.softmax(pre_out_j, 2).squeeze(1) # this necessary?
            max_values, max_idxs = probs_j.max(1)
            sequences = torch.cat((sequences, max_idxs.unsqueeze(1)), 1)
        return sequences
    # Maybe code greedy first
    # TODO calc batch-wise.....really slow
    # def beam_search(self, x, x_mask_0, z, max_len, beam_size=3):
    #     # batch_size = x.shape[0]
    #     f = self.emb_x(x.long())
    #     bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
    #     s_0, _ = self.rnn_bigru_x(f, bigru_x_h0)
    #
    #     t = [torch.tanh(self.aff_init_dec(z))]
    #     proj_key_0 = self.attention.key_layer(s_0)
    #     prev_beams = [[[self.sos_idx], 0, 0, False]]
    #     for j in range(max_len):
    #         y_j1 = []
    #         t_j1 = []
    #         done_beams = []
    #         for beam_idx, beam in enumerate(prev_beams):
    #             if beam[3]:
    #                 done_beams.append(beam)
    #             else:
    #                 y_j1.append(beam[0][-1])
    #                 t_j1.append(t[j][beam[2]])
    #
    #         y_j1 = torch.unsqueeze(torch.tensor(y_j1), 1).to(self.device)
    #         e_j = self.emb_y(y_j1.long())
    #
    #
    #         t_j1 = torch.stack(t_j1).to(self.device)
    #
    #         proj_key = proj_key_0.expand(t_j1.shape[0] ,-1,-1)
    #         s = s_0.expand(t_j1.shape[0] ,-1,-1)
    #         x_mask = x_mask_0.expand(t_j1.shape[0] ,-1,-1)
    #         c_j, _ = self.attention(t_j1.unsqueeze(1), proj_key, s, x_mask_0)
    #         # pri
    #         h0 = torch.cat((c_j, e_j), 2)
    #         t_j, _ = self.rnn_gru_dec(t_j1.unsqueeze(1), torch.squeeze(h0, 1).unsqueeze(0)) # Changed due no batch
    #
    #         t.append(torch.squeeze(t_j, 1))
    #
    #         pre_out_j = self.aff_out_y(torch.cat((t_j, e_j), 2))
    #         probs_j = F.softmax(pre_out_j, 2).squeeze(1)
    #
    #         new_beams = []
    #         # Init with done beams
    #         for i in range(probs_j.shape[0]):
    #             for w in range(probs_j.shape[1]):
    #                 prev_beam = prev_beams[i]
    #                 done = False
    #                 if w == self.eos_idx:
    #                     done = True
    #                 new_beam = [
    #                     prev_beam[0] + [i],
    #                     prev_beam[1] + torch.log(probs_j[i][w]).item(),
    #                     i,
    #                     done
    #                 ]
    #                 if len(new_beams) < beam_size:
    #                     new_beams.append(new_beam)
    #                     new_beams.sort(key=lambda x: x[1], reverse=True)
    #                 else:
    #                     if new_beams[-1][1] < new_beam[1]:
    #                         new_beams[-1] = new_beam
    #                         new_beams.sort(key=lambda x: x[1], reverse=True)
    #         prev_beams = new_beams

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
