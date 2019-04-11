import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import copy
import numpy as np

class AEVNMT(nn.Module):
    def __init__(self, vocab, vocab_size, emb_dim, padding_idx, hidden_dim, max_len, device, train=False, sos_idx=None, eos_idx=None, pad_idx=None):
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
        self.source = SourceModel(self.vocab, self.emb_x, emb_dim, hidden_dim, vocab_size, train=train, sos_idx=sos_idx, eos_idx=eos_idx).to(device)
        self.trans = TransModel(self.vocab, self.emb_x, self.emb_y, emb_dim, hidden_dim, vocab_size, self.attention, device, train=train, sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx).to(device)
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
    def __init__(self, vocab, emb_x, emb_dim, hidden_dim, vocab_size, train=False, sos_idx=None, eos_idx=None):
        super(SourceModel, self).__init__()
        self.sos_idx = sos_idx
        self.train = train
        self.vocab = vocab
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
        # print("Source model")
        for j in range(max_len):
            # print("Timestep j: ", j)
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
    def __init__(self, vocab, emb_x, emb_y, emb_dim, hidden_dim, vocab_size, attention, device, train=False, sos_idx=None, eos_idx=None, pad_idx=None):
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

        self.attention = attention

        self.aff_init_dec = nn.Linear(hidden_dim, self.t_size)
        self.rnn_gru_dec = nn.GRU(self.t_size, self.t_size, batch_first=True)

        self.aff_out_y = nn.Linear(self.t_size + emb_dim, vocab_size)
        self.train = train
        self.device = device

    def forward(self, x, x_mask, z, y=None, max_len=None):
        # print("-- Train")
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


    # def beam_decode(self, x, x_mask, z, max_len, beam_size=3):
    #     batch_size = x.shape[0]
    #
    #     f = self.emb_x(x.long())
    #     bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
    #     s, _ = self.rnn_bigru_x(f, bigru_x_h0)
    #
    #     t = [torch.tanh(self.aff_init_dec(z))]
    #     proj_key = self.attention.key_layer(s)
    #
    #     print("x: ", x.shape)

        # beams = [[torch.tensor([self.sos_idx])] for _ in range(batch_size)]

        # asd

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
            # sequences = new_sequences
                # print(batch)
                # print(batch)

            # asd

            # print(new_sequences)
            # print(new_log_probs)
            # ASD
                # asd

                # new_beams
            # print(new_sequences)
            # print(new_log_probs)
            # asd
                # print(sequences[batch][beam])
                # print("==========")
                # asd
                # sequence =


            #
            #     beams = []
            #     beam_probs = []
            #     for b in range(beam_size):
            #         beams.append(sequences[batch][beam] + [idx[i][b].item()])
            #         beam_probs.append(log_probs[batch][beam] + values[i][b].item())
            #     sequences[batch].extend(new_beams)
                # print(sequences[batch][beam])
                # print(new_beams)
                # print(new_probs)

                # asd
                    # new_seq = sequence[batch][beam].append()
                    # prev_seq = sequences[batch][beam]
                    # new_seq = torch.cat((sequences[batch][beam].to(), idx[i]))
                    # print(new_seq)
                    # prev_seq.candidate
                    # asd

                # prev_seqs = [sequences[batch][beam] for _ in range(beam_size)]
                # for b in range(range(beam_size)):
                # print(prev_seqs)
                # prev_seqs = sequences[batch][beam].unsqueeze(1).expand(beam_size, -1).to(self.device)
                # new_seqs = torch.cat((prev_seqs, idx[i].unsqueeze(1)), 1)


                # print(new_seqs)
                # print("prev_seq", )
                # print(batch, beam)
                # print(idx[i])
                # print(values[i])
                #
                # print("prev_seq", prev_seq.unsqueeze(1).expand(beam_size, -1))
                # print(batch, beam)
                # print(idx[i])
                # print(values[i])


            # print

            # print("partial_idxs: ", partial_idxs)
            # print("log_probs: ", log_probs)
            # print("idx: ", idx)
            # print("values: ", values)
            # probs_expanded = log_probs.unsqueeze(2).expand(-1, -1, beam_size) # check how this works after beam is already 3..
            # new_probs = torch.zeros(probs_expanded.shape).to(self.device)

    #         print(new_probs)
            # for i, (batch, beam) in enumerate(partial_idxs):
            #     new_probs[batch, beam] += values[i]
            #     print(idx[i])
                # old_seq = sequences[batch][beam]
                # new_seqs = [torch.cat((sequences[batch][beam], idx[i][k])) for k in range(beam_size)]
                # print(new_seqs)
                # for j in range()
                # print(new_seqs)
                # print(idx[i])
                # sequences[batch][beam] += idx[i]
                # print("sequences: ", sequences[batch][beam])
            # print(new_probs)
                # print(values[i])
                # new_probs[batch, beam] += values[batch, beam]
            # print(new_probs)
                # print(probs_expanded)
                # print("sequences: ", sequences)
                #
                # print(probs_expanded[batch, beam])
                # print(batch, beam)
                #
                #
                # print("idx: ", idx)
                # print("idx: ", idx.shape)
                #
                # print("values: ", values)
                # print("values: ", values.shape)






            # for batch, idx in partial_idxs:
                #

            # asd
            # sequences = torch.cat((probs_expanded, idx), 2)



            # print("log_prob_exps: " , probs_expanded.shape)
            # print("values: ", values.shape)
            # print("values: ", values)
            # print("log_probs: ", log_probs)



            # print("sequences: ",  sequences)
            # seq_expanded = sequences.expand(-1, 3, -1)
            # sequences = torch.cat((seq_expanded, idx), 2) # should be fine now


            # log_probs
            # log_probs = torch.cat((log_probs, probs_expanded), 2)
            # print(log_probs)
            # print("values: ", values)
            # print(values[-5:], idx[-5:])

        # print(sequences)

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
