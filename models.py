import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import copy
import numpy as np

class AEVNMT(nn.Module):
    def __init__(self, vocab, vocab_size, emb_dim, padding_idx, hidden_dim, max_len, device, dropout=0.3, sos_idx=None, eos_idx=None, pad_idx=None):
        super(AEVNMT, self).__init__()

        self.device = device
        self.vocab = vocab
        # Initialize parameters
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.dropout = dropout

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
        self.source = SourceModel(self.vocab, self.emb_x, emb_dim, hidden_dim, vocab_size, dropout, sos_idx=sos_idx, eos_idx=eos_idx).to(device)
        self.trans = TransModel(self.vocab, self.emb_x, self.emb_y, emb_dim, hidden_dim, vocab_size, self.attention, device, dropout, sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx).to(device)
        self.enc = SentEmbInfModel(self.emb_x, emb_dim, hidden_dim).to(device)

    def forward(self, x, x_mask, y=None, y_mask=None):
        mu_theta, sigma_theta = self.enc.forward(x)

        batch_size = x.shape[0]
        e = self.normal.sample(sample_shape=torch.tensor([batch_size])).to(self.device)
        z = mu_theta + e * (sigma_theta ** 2)

        pre_out_x = self.source.forward(z, x=x)
        pre_out_y = self.trans.forward(x, x_mask, z, y=y)

        return pre_out_x, pre_out_y, mu_theta, sigma_theta

    def predict(self, x, x_mask):
        with torch.no_grad():
            mu_theta, sigma_theta = self.enc.forward(x)
            # predictions = self.trans.beam_decode(x, x_mask, mu_theta, self.max_len)
            predictions = self.trans.greedy_decode(x, x_mask, mu_theta, self.max_len)
            # print(predictions)
            return predictions
            # print(mu_theta.shape)

class SourceModel(nn.Module):
    def __init__(self, vocab, emb_x, emb_dim, hidden_dim, vocab_size, dropout, sos_idx=None, eos_idx=None):
        super(SourceModel, self).__init__()
        self.sos_idx = sos_idx
        self.vocab = vocab
        self.emb_x = emb_x

        # self.tanh = nn.Tanh()
        self.aff_init_lm = nn.Linear(hidden_dim, hidden_dim)
        self.rnn_gru_lm = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.aff_out_x = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, x=None, max_len=None):
        if max_len == None:
            max_len = x.shape[-1]

        pre_out = []

        h = [torch.tanh(self.aff_init_lm(z))]
        for j in range(max_len):
            f_j = self.emb_x(torch.unsqueeze(x[:, j], 1).long())
            h_j, _ = self.rnn_gru_lm(f_j, h[j].unsqueeze(0))
            h_j = self.dropout(h_j)
            h.append(torch.squeeze(h_j))
            pre_out.append(self.aff_out_x(h[j]))
        return pre_out

# Non-continious
class TransModel(nn.Module):
    def __init__(self, vocab, emb_x, emb_y, emb_dim, hidden_dim, vocab_size, attention, device, dropout=0.3, sos_idx=None, eos_idx=None, pad_idx=None):
        super(TransModel, self).__init__()
        self.vocab = vocab
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.emb_x = emb_x
        self.emb_y = emb_y
        self.t_size = hidden_dim * 2 + emb_dim

        self.aff_init_enc = nn.Linear(hidden_dim, hidden_dim)
        self.rnn_bigru_x = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.attention = attention

        self.aff_init_dec = nn.Linear(hidden_dim, self.t_size)
        self.rnn_gru_dec = nn.GRU(self.t_size, self.t_size, batch_first=True)

        self.aff_out_y = nn.Linear(self.t_size + emb_dim, vocab_size)
        self.device = device

    def forward(self, x, x_mask, z, y=None, max_len=None):
        batch_size = x.shape[0]

        f = self.emb_x(x.long())
        bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
        s, _ = self.rnn_bigru_x(f, bigru_x_h0)
        s = self.dropout(s)

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
            t_j = self.dropout(t_j)
            t.append(torch.squeeze(t_j))
            pre_out.append(self.aff_out_y(torch.cat((t_j, e_j), 2)))
        return pre_out

    def greedy_decode(self, x, x_mask, z, max_len):
        # print("-- Decode")
        batch_size = x.shape[0]

        f = self.emb_x(x.long())
        bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
        s, _ = self.rnn_bigru_x(f, bigru_x_h0)
        s = self.dropout(s)

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
            t_j = self.dropout(t_j)
            t.append(torch.squeeze(t_j))
            pre_out_j = self.aff_out_y(torch.cat((t_j, e_j), 2))
            probs_j = F.softmax(pre_out_j, 2).squeeze(1) # this necessary?
            max_values, max_idxs = probs_j.max(1)
            sequences = torch.cat((sequences, max_idxs.unsqueeze(1)), 1)
        return sequences

    # Node based???
    def beam_decode(self, x, x_mask, z, max_len, beam_size = 3):
        batch_size = x.shape[0]

        f = self.emb_x(x.long())
        bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
        s, _ = self.rnn_bigru_x(f, bigru_x_h0)

        t = [torch.tanh(self.aff_init_dec(z))]
        proj_key = self.attention.key_layer(s)

        done = [[0] for _ in range(batch_size)]
        log_probs = [[0] for _ in range(batch_size)]
        sequences = [[[self.sos_idx]] for _ in range(batch_size)]

        for j in range(max_len):
            partial_idxs = np.transpose(np.argwhere(torch.tensor(done)==0))
            done_idxs = np.transpose(np.argwhere(torch.tensor(done)==1))
            input = None
            proj_key_input = None
            t_j_input = None
            x_mask_input = None
            s_input = None
            # print()
            batches_idx = []
            if partial_idxs.shape[0] == 0:
                break

            for batch, beam in partial_idxs:
                if batch not in batches_idx:
                    batches_idx.append(batch)
                batch_idx = batches_idx.index(batch)
                # batches_idx.append(batch)

                sequence = torch.tensor([sequences[batch][beam][j]]).unsqueeze(0)
                if input is None:
                    input = sequence # slice to keep dim
                else:
                    input = torch.cat((input, sequence), 0)

                if proj_key_input is None:
                    proj_key_input = proj_key[batch:batch+1, :, :]
                else:
                    proj_key_input = torch.cat((proj_key_input, proj_key[batch:batch+1, :, :]), 0)

                if t_j_input is None:
                    t_j_input = t[j][batch_idx:batch_idx+1, :]
                else:
                    t_j_input = torch.cat((t_j_input, t[j][batch_idx:batch_idx+1, :]), 0)

                if x_mask_input is None:
                    x_mask_input = x_mask[batch:batch+1, :]
                else:
                    x_mask_input = torch.cat((x_mask_input, x_mask[batch:batch+1, :]), 0)

                if s_input is None:
                    s_input = s[batch:batch+1, :, :]
                else:
                    s_input = torch.cat((s_input, s[batch:batch+1, :, :]), 0)

            e_j = self.emb_y(input.to(self.device))
            t_j_input = t_j_input.unsqueeze(1)
            c_j, _ = self.attention(t_j_input, proj_key_input, s_input, x_mask_input)

            h0 = torch.cat((c_j, e_j), 2)
            t_j, _ = self.rnn_gru_dec(t_j_input, h0.squeeze(1).unsqueeze(0))
            t.append(t_j.squeeze(1))
            pre_out_j = self.aff_out_y(torch.cat((t_j, e_j), 2))
            log_probs_j = torch.log(F.softmax(pre_out_j, 2).squeeze(1))
            max_probs, max_idx = torch.sort(log_probs_j, 1)
            idx = torch.narrow(max_idx, 1, -beam_size, beam_size)
            values = torch.narrow(max_probs, 1, -beam_size, beam_size)

            new_sequences = [[] for _ in range(batch_size)]
            new_log_probs = [[] for _ in range(batch_size)]
            new_done = [[] for _ in range(batch_size)]
            for i, (batch, beam) in enumerate(done_idxs):
                new_sequences[batch].append(sequences[batch][beam] + [self.pad_idx])
                new_log_probs[batch].append(log_probs[batch][beam])
                new_done[batch].append(done[batch][beam])

            for i, (batch, beam) in enumerate(partial_idxs):
                new_beams = []
                new_probs = []
                new_dones = []
                for b in range(beam_size):
                    new_beams.append(sequences[batch][beam] + [idx[i][b].item()])
                    new_probs.append(log_probs[batch][beam] + values[i][b].item())
                    if idx[i][b].item() == self.eos_idx:
                        new_dones.append(1)
                    else:
                        new_dones.append(0)
                        # print
                new_sequences[batch].extend(new_beams)
                new_log_probs[batch].extend(new_probs)
                new_done[batch].extend(new_dones)

            # Cutoff at top beam-size
            for b in range(batch_size):
                batch = new_log_probs[b]
                # print("batch: ", batch)
                sorted_batch = sorted(range(len(batch)), key=lambda k: batch[k])
                top_idx = sorted_batch[-beam_size:]
                top_sequences = [new_sequences[b][i] for i in top_idx]
                top_log_probs = [new_log_probs[b][i] for i in top_idx]
                top_done = [new_done[b][i] for i in top_idx]
                new_sequences[b] = top_sequences
                new_log_probs[b] = top_log_probs
                new_done[b] = top_done
            sequences = new_sequences
            log_probs = new_log_probs
            done = new_done
            # print(sequences)

        # Get only the top
        for b in range(batch_size):
            batch = log_probs[b]
            idx = np.argmax(batch)
            sequences[b] = sequences[b][idx]
        return torch.tensor(sequences)

# Currently is not conditioned on the target sentence (should be an option)
class SentEmbInfModel(nn.Module):
    def __init__(self, emb_x, emb_dim, hidden_dim, dropout=0.3):
        super(SentEmbInfModel, self).__init__()
        self.emb_x = emb_x

        # TODO: init GRU cells to zero (page 44, thesis)
        self.rnn_gru_x = nn.GRU(emb_dim, hidden_dim, batch_first = True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.aff_u_hid = nn.Linear(2 * hidden_dim, hidden_dim)
        self.aff_u_out = nn.Linear(hidden_dim, hidden_dim)

        self.aff_s_hid = nn.Linear(2 * hidden_dim, hidden_dim)
        self.aff_s_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        f = self.emb_x(x.long())
        f.detach()
        gru_x, w = self.rnn_gru_x(f)
        gru_x = self.dropout(gru_x)
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
