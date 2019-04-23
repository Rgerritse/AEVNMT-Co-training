import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import copy
import numpy as np
from joeynmt.helpers import tile


class AEVNMT(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, config):
        super(AEVNMT, self).__init__()
        # self.device = device
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

        # print("vocab_src", vocab_src.stoi["<s>"])
        # asd

        # Initialize priors
        self.mu_prior = torch.tensor([0.0] * config["hidden_dim"])
        self.sigma_prior = torch.tensor([1.0] * config["hidden_dim"])
        self.normal = Normal(self.mu_prior, self.sigma_prior)

        # Initialize embeddings
        self.emb_x = nn.Embedding(len(vocab_src), config["emb_dim"], padding_idx=vocab_src.stoi[config["pad"]])
        self.emb_y = nn.Embedding(len(vocab_tgt), config["emb_dim"], padding_idx=vocab_tgt.stoi[config["pad"]])

        # Misc
        self.model_type = config["model_type"]
        self.config = config
        self.device =  torch.device(config["device"])

        # Initialize models
        self.attention = BahdanauAttention(config["hidden_dim"], query_size = 2 * config["hidden_dim"] + config["emb_dim"])
        self.source = SourceModel(vocab_src, self.emb_x, config).to(self.device)
        self.trans = TransModel(vocab_tgt, self.emb_x, self.emb_y, self.attention, config).to(self.device)
        self.enc = SentEmbInfModel(self.emb_x, config).to(self.device)
        # self.enc = SentEmbInfModel(self.emb_x, emb_dim, hidden_dim).to(device)

    def forward(self, x, x_mask, y=None, y_mask=None):
        mu_theta, sigma_theta = self.enc.forward(x)

        if self.model_type == "nmt":
            z = mu_theta
        else:
            raise ValueError("Invalid model type")

        # batch_size = x.shape[0]
        # e = self.normal.sample(sample_shape=torch.tensor([batch_size])).to(self.device)
        # z = mu_theta + e * (sigma_theta ** 2)

        pre_out_x = self.source.forward(z, x=x)
        pre_out_y = self.trans.forward(x, x_mask, z, y=y)

        return pre_out_x, pre_out_y, mu_theta, sigma_theta

    def predict(self, x, x_mask):
        with torch.no_grad():
            mu_theta, _ = self.enc.forward(x)
            predictions = self.trans.beam_search(x, x_mask, mu_theta)
            return predictions
            # print(mu_theta.shape)

class SourceModel(nn.Module):
    def __init__(self, vocab_src, emb_x, config):
        super(SourceModel, self).__init__()
        self.vocab = vocab_src
        self.emb_x = emb_x

        # self.tanh = nn.Tanh()
        self.aff_init_lm = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.rnn_gru_lm = nn.GRU(config["emb_dim"], config["hidden_dim"], batch_first=True)
        self.aff_out_x = nn.Linear(config["hidden_dim"], len(vocab_src))
        self.dropout = nn.Dropout(config["dropout"])

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
    def __init__(self, vocab_tgt, emb_x, emb_y, attention, config):
        super(TransModel, self).__init__()
        self.vocab_tgt = vocab_tgt
        self.sos_idx = vocab_tgt.stoi[config["sos"]]
        self.eos_idx = vocab_tgt.stoi[config["eos"]]
        self.pad_idx = vocab_tgt.stoi[config["pad"]]

        self.vocab_size = len(vocab_tgt)
        # self.eos_idx = eos_idx
        # self.pad_idx = pad_idx
        self.emb_x = emb_x
        self.emb_y = emb_y
        self.t_size = config["hidden_dim"] * 2 + config["emb_dim"]

        self.aff_init_enc = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.rnn_bigru_x = nn.GRU(config["emb_dim"], config["hidden_dim"], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config["dropout"])
        self.word_dropout = nn.Dropout(config["word_dropout"])

        self.attention = attention

        self.aff_init_dec = nn.Linear(config["hidden_dim"], self.t_size)
        self.rnn_gru_dec = nn.GRU(self.t_size, self.t_size, batch_first=True)

        self.aff_out_y = nn.Linear(self.t_size + config["emb_dim"], len(vocab_tgt))

        self.config = config
        self.device =  torch.device(config["device"])

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
                # print("torch.unsqueeze(y[:, j], 1: ", torch.unsqueeze(y[:, j], 1).long().shape)
                e_j = self.emb_y(torch.unsqueeze(y[:, j], 1).long())
            # print("e_j.shape: ", e_j.shape)
            e_j = self.word_dropout(e_j)


            h0 = torch.cat((c_j, e_j), 2)
            t_j, _ = self.rnn_gru_dec(t[j].unsqueeze(1), torch.squeeze(h0).unsqueeze(0))
            t_j = self.dropout(t_j)
            t.append(torch.squeeze(t_j))
            pre_out.append(self.aff_out_y(torch.cat((t_j, e_j), 2)))
        return pre_out

    # Based on joeynmt beam search
    def beam_search(self, x, x_mask, z, n_best = 1):
        # print(self.vocab_tgt.string(x))

        size = self.config["beam_size"]
        batch_size = x_mask.shape[0]

        f = self.emb_x(x.long())
        bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
        s, _ = self.rnn_bigru_x(f, bigru_x_h0)
        s = self.dropout(s).contiguous()

        t_j = torch.tanh(self.aff_init_dec(z))
        proj_key = self.attention.key_layer(s)

        # Tile attention inputs
        t_j = tile(t_j, size, dim=0)
        proj_key = tile(proj_key, size, dim=0)
        s = tile(s, size, dim=0)
        src_mask = tile(x_mask, size, dim=0)

        batch_offset = torch.arange(batch_size, dtype=torch.long, device=self.device)
        beam_offset = torch.arange(
            0,
            batch_size * size,
            step=size,
            dtype=torch.long,
            device=self.device
        )
        alive_seq = torch.full(
            [batch_size * size, 1],
            self.sos_idx,
            dtype=torch.long,
            device=self.device
        )

        # print("batch_offset: ", batch_offset)
        # print("beam_offset: ", beam_offset)
        # print("alive_seq: ", alive_seq)

        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (size - 1),
                                   device=self.device).repeat(
                                    batch_size))

        # print("topk_log_probs: ", topk_log_probs)

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["gold_score"] = [0] * batch_size

        for step in range(self.config["max_len"]):
            # print("step: ", step)
            decoder_input = alive_seq[:, -1].view(-1, 1)
            e_j = self.emb_y(decoder_input)
            # print("t_j: ", t_j.shape)
            # print("proj_key: ", proj_key.shape)
            # print("s: ", s.shape)
            # print("src_mask: ", src_mask.shape)

            c_j, _ = self.attention(t_j.unsqueeze(1), proj_key, s, src_mask)

            h0 = torch.cat((c_j, e_j), 2)
            t_j, _ = self.rnn_gru_dec(t_j.unsqueeze(1), torch.squeeze(h0).unsqueeze(0))
            t_j = self.dropout(t_j)
            out = self.aff_out_y(torch.cat((t_j, e_j), 2))
            t_j = t_j.squeeze(1)

            log_probs = F.log_softmax(out, dim=-1).squeeze(1)
            # print("log_probs: ", log_probs)

            # multiply probs by the beam probability (=add logprobs)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            curr_scores = log_probs

            # compute length penalty
            if self.config["length_penalty"] > -1:
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** self.config["length_penalty"]
                curr_scores /= length_penalty

            # flatten log_probs into a list of possibilities
            curr_scores = curr_scores.reshape(-1, size * self.vocab_size)

            # pick currently best top k hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

            if self.config["length_penalty"] > -1:
                # recover original log probs
                topk_log_probs = topk_scores * length_penalty

             # reconstruct beam origin and true word ids from flattened order
            topk_beam_index = topk_ids.div(self.vocab_size)
            topk_ids = topk_ids.fmod(self.vocab_size)

            # map beam_index to batch_index in the flat representation
            batch_index = (
                topk_beam_index
                + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # append latest prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

            is_finished = topk_ids.eq(self.eos_idx)
            if step + 1 == self.config["max_len"]:
                is_finished.fill_(1)
            # end condition is whether the top beam is finished
            end_condition = is_finished[:, 0].eq(1)

             # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # store finished hypotheses for this batch
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:])  # ignore start_token
                        )
                    # if the batch reached the end, save the n_best hypotheses
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                 # if all sentences are translated, no need to go further
                # pylint: disable=len-as-condition
                if len(non_finished) == 0:
                    break
                # remove finished batches for the next step
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

                # reorder indices, outputs and masks
                select_indices = batch_index.view(-1)
                # encoder_output = encoder_output.index_select(0, select_indices)
                t_j = t_j.index_select(0, select_indices)
                proj_key = proj_key.index_select(0, select_indices)
                s = s.index_select(0, select_indices)
                src_mask = src_mask.index_select(0, select_indices)

                # if isinstance(hidden, tuple):
                #     # for LSTMs, states are tuples of tensors
                #     h, c = hidden
                #     h = h.index_select(1, select_indices)
                #     c = c.index_select(1, select_indices)
                #     hidden = (h, c)
                # else:
                #     # for GRUs, states are single tensors
                #     hidden = hidden.index_select(1, select_indices)
                #
                # att_vectors = att_vectors.index_select(0, select_indices)
        def pad_and_stack_hyps(hyps, pad_value):
            filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
            for j, h in enumerate(hyps):
                for k, i in enumerate(h):
                    filled[j, k] = i
            return filled

        # for pred in results["predictions"]:
        #     print(self.vocab_tgt.string(pred[0]))
            # print(self.vocab_tgt.string(pred))
        # print(results["predictions"])
        # print(self.vocab_tgt.string(torch.tensor(results["predictions"])))
        # print("sequences: ", self.vocab_tgt.string(torch.tensor(results["predictions"])))
        # print(results["predictions"])
        final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=self.pad_idx)
        return final_outputs


    # def greedy_decode(self, x, x_mask, z):
    #     # print("-- Decode")
    #     batch_size = x.shape[0]
    #
    #     f = self.emb_x(x.long())
    #     bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
    #     s, _ = self.rnn_bigru_x(f, bigru_x_h0)
    #     s = self.dropout(s)
    #
    #     t = [torch.tanh(self.aff_init_dec(z))]
    #     proj_key = self.attention.key_layer(s)
    #
    #     sequences = torch.tensor([[self.sos_idx] for _ in range(batch_size)]).to(self.device)
    #     for j in range(self.config["max_len"]):
    #         # print("Timestep: ", j)
    #         # print("Input: \n", self.vocab.string(torch.unsqueeze(sequences[:, j], 1)))
    #         c_j, _ = self.attention(t[j].unsqueeze(1), proj_key, s, x_mask)
    #         e_j = self.emb_y(torch.unsqueeze(sequences[:, j], 1).long())
    #
    #         h0 = torch.cat((c_j, e_j), 2)
    #         t_j, _ = self.rnn_gru_dec(t[j].unsqueeze(1), torch.squeeze(h0).unsqueeze(0))
    #         t_j = self.dropout(t_j)
    #         t.append(torch.squeeze(t_j))
    #         pre_out_j = self.aff_out_y(torch.cat((t_j, e_j), 2))
    #         probs_j = F.softmax(pre_out_j, 2).squeeze(1) # this necessary?
    #         max_values, max_idxs = probs_j.max(1)
    #         sequences = torch.cat((sequences, max_idxs.unsqueeze(1)), 1)
    #     return sequences

    # Node based???
    # def beam_decode(self, x, x_mask, z, max_len, beam_size = 3):
    #     batch_size = x.shape[0]
    #
    #     f = self.emb_x(x.long())
    #     bigru_x_h0 = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()
    #     s, _ = self.rnn_bigru_x(f, bigru_x_h0)
    #
    #     t = [torch.tanh(self.aff_init_dec(z))]
    #     proj_key = self.attention.key_layer(s)
    #
    #     done = [[0] for _ in range(batch_size)]
    #     log_probs = [[0] for _ in range(batch_size)]
    #     sequences = [[[self.sos_idx]] for _ in range(batch_size)]
    #
    #     for j in range(max_len):
    #         partial_idxs = np.transpose(np.argwhere(torch.tensor(done)==0))
    #         done_idxs = np.transpose(np.argwhere(torch.tensor(done)==1))
    #         input = None
    #         proj_key_input = None
    #         t_j_input = None
    #         x_mask_input = None
    #         s_input = None
    #         # print()
    #         batches_idx = []
    #         if partial_idxs.shape[0] == 0:
    #             break
    #
    #         for batch, beam in partial_idxs:
    #             if batch not in batches_idx:
    #                 batches_idx.append(batch)
    #             batch_idx = batches_idx.index(batch)
    #             # batches_idx.append(batch)
    #
    #             sequence = torch.tensor([sequences[batch][beam][j]]).unsqueeze(0)
    #             if input is None:
    #                 input = sequence # slice to keep dim
    #             else:
    #                 input = torch.cat((input, sequence), 0)
    #
    #             if proj_key_input is None:
    #                 proj_key_input = proj_key[batch:batch+1, :, :]
    #             else:
    #                 proj_key_input = torch.cat((proj_key_input, proj_key[batch:batch+1, :, :]), 0)
    #
    #             if t_j_input is None:
    #                 t_j_input = t[j][batch_idx:batch_idx+1, :]
    #             else:
    #                 t_j_input = torch.cat((t_j_input, t[j][batch_idx:batch_idx+1, :]), 0)
    #
    #             if x_mask_input is None:
    #                 x_mask_input = x_mask[batch:batch+1, :]
    #             else:
    #                 x_mask_input = torch.cat((x_mask_input, x_mask[batch:batch+1, :]), 0)
    #
    #             if s_input is None:
    #                 s_input = s[batch:batch+1, :, :]
    #             else:
    #                 s_input = torch.cat((s_input, s[batch:batch+1, :, :]), 0)
    #
    #         e_j = self.emb_y(input.to(self.device))
    #         t_j_input = t_j_input.unsqueeze(1)
    #         c_j, _ = self.attention(t_j_input, proj_key_input, s_input, x_mask_input)
    #
    #         h0 = torch.cat((c_j, e_j), 2)
    #         t_j, _ = self.rnn_gru_dec(t_j_input, h0.squeeze(1).unsqueeze(0))
    #         t.append(t_j.squeeze(1))
    #         pre_out_j = self.aff_out_y(torch.cat((t_j, e_j), 2))
    #         log_probs_j = torch.log(F.softmax(pre_out_j, 2).squeeze(1))
    #         max_probs, max_idx = torch.sort(log_probs_j, 1)
    #         idx = torch.narrow(max_idx, 1, -beam_size, beam_size)
    #         values = torch.narrow(max_probs, 1, -beam_size, beam_size)
    #
    #         new_sequences = [[] for _ in range(batch_size)]
    #         new_log_probs = [[] for _ in range(batch_size)]
    #         new_done = [[] for _ in range(batch_size)]
    #         for i, (batch, beam) in enumerate(done_idxs):
    #             new_sequences[batch].append(sequences[batch][beam] + [self.pad_idx])
    #             new_log_probs[batch].append(log_probs[batch][beam])
    #             new_done[batch].append(done[batch][beam])
    #
    #         for i, (batch, beam) in enumerate(partial_idxs):
    #             new_beams = []
    #             new_probs = []
    #             new_dones = []
    #             for b in range(beam_size):
    #                 new_beams.append(sequences[batch][beam] + [idx[i][b].item()])
    #                 new_probs.append(log_probs[batch][beam] + values[i][b].item())
    #                 if idx[i][b].item() == self.eos_idx:
    #                     new_dones.append(1)
    #                 else:
    #                     new_dones.append(0)
    #                     # print
    #             new_sequences[batch].extend(new_beams)
    #             new_log_probs[batch].extend(new_probs)
    #             new_done[batch].extend(new_dones)
    #
    #         # Cutoff at top beam-size
    #         for b in range(batch_size):
    #             batch = new_log_probs[b]
    #             # print("batch: ", batch)
    #             sorted_batch = sorted(range(len(batch)), key=lambda k: batch[k])
    #             top_idx = sorted_batch[-beam_size:]
    #             top_sequences = [new_sequences[b][i] for i in top_idx]
    #             top_log_probs = [new_log_probs[b][i] for i in top_idx]
    #             top_done = [new_done[b][i] for i in top_idx]
    #             new_sequences[b] = top_sequences
    #             new_log_probs[b] = top_log_probs
    #             new_done[b] = top_done
    #         sequences = new_sequences
    #         log_probs = new_log_probs
    #         done = new_done
    #         # print(sequences)
    #
    #     # Get only the top
    #     for b in range(batch_size):
    #         batch = log_probs[b]
    #         idx = np.argmax(batch)
    #         sequences[b] = sequences[b][idx]
    #     return torch.tensor(sequences)

# Currently is not conditioned on the target sentence (should be an option)
class SentEmbInfModel(nn.Module):
    def __init__(self, emb_x, config):
        super(SentEmbInfModel, self).__init__()
        self.emb_x = emb_x

        # TODO: init GRU cells to zero (page 44, thesis)
        self.rnn_gru_x = nn.GRU(config["emb_dim"], config["hidden_dim"], batch_first = True, bidirectional=True)
        self.dropout = nn.Dropout(config["dropout"])

        self.aff_u_hid = nn.Linear(2 * config["hidden_dim"], config["hidden_dim"])
        self.aff_u_out = nn.Linear(config["hidden_dim"], config["hidden_dim"])

        self.aff_s_hid = nn.Linear(2 * config["hidden_dim"], config["hidden_dim"])
        self.aff_s_out = nn.Linear(config["hidden_dim"], config["hidden_dim"])

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
