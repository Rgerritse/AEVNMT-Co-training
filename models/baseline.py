import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from joeynmt.attention import BahdanauAttention, LuongAttention
from joeynmt.helpers import tile
import numpy as np

class Baseline(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, config):
        super(Baseline, self).__init__()

        self.config = config

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

        # Embeddings with gaussian initialization
        self.emb_x = nn.Embedding(len(vocab_src), config["emb_dim"], padding_idx=vocab_src.stoi[config["pad"]])
        self.emb_y = nn.Embedding(len(vocab_tgt), config["emb_dim"], padding_idx=vocab_tgt.stoi[config["pad"]])
        nn.init.normal_(self.emb_x.weight, std=config["emb_init_std"])
        nn.init.normal_(self.emb_y.weight, std=config["emb_init_std"])

        if config["attention"] == "bahdanau":
            self.attention = BahdanauAttention(config["hidden_dim"], 2 * config["hidden_dim"], config["hidden_dim"])
        elif config["attention"] == "luong":
            self.attention = LuongAttention(config["hidden_dim"], 2 * config["hidden_dim"])
        self.inference = InferenceModel(self.emb_x, config)
        self.encoder = Encoder(self.emb_x, config)
        self.decoder = Decoder(vocab_tgt, self.emb_y, self.attention, config)

    def forward(self, x, x_mask, prev, prev_mask, y):
        mu, sigma = self.inference(x)
        if self.config["model_type"] == "nmt":
            z = mu
        elif self.config["model_type"] == "aevnmt":
            batch_size = x.shape[0]
            e = self.normal.sample(sample_shape=torch.tensor([batch_size])).to(self.device)
            z = mu_theta + e * sigma_theta
        else:
            raise ValueError("Invalid model type")

        enc_output, enc_hidden = self.encoder.forward(x, z)
        logits_y_vectors = self.decoder.forward(enc_output, enc_hidden, x_mask, prev, z)

        if self.config["model_type"] == "nmt":
            loss = self.compute_nmt_loss(logits_y_vectors, y)

        return logits_y_vectors, loss

    def predict(self, x, x_mask):
        with torch.no_grad():
            mu, sigma = self.inference(x)
            if self.config["model_type"] == "nmt":
                z = mu
            else:
                raise ValueError("Invalid model type")
            enc_output, enc_hidden = self.encoder.forward(x, z)
            predictions = self.decoder.predict(enc_output, enc_hidden, x_mask, z)
            return predictions

    def compute_nmt_loss(self, logits_y, y):
        loss = F.cross_entropy(
            logits_y.view(-1, len(self.vocab_tgt)),
            y.long().view(-1),
            ignore_index=self.vocab_tgt.stoi[self.config["pad"]],
            reduction="mean")
        return loss

class InferenceModel(nn.Module):
    def __init__(self, emb_x, config):
        super(InferenceModel, self).__init__()
        self.emb_x = emb_x

        self.rnn_gru_x = nn.GRU(config["emb_dim"], config["hidden_dim"], batch_first = True, bidirectional=True)
        # init this gru with xavier

        self.dropout = nn.Dropout(config["dropout"])

        self.aff_u_hid = nn.Linear(2 * config["hidden_dim"], config["hidden_dim"])
        self.aff_u_out = nn.Linear(config["hidden_dim"], config["hidden_dim"])

        self.aff_s_hid = nn.Linear(2 * config["hidden_dim"], config["hidden_dim"])
        self.aff_s_out = nn.Linear(config["hidden_dim"], config["hidden_dim"])

    def forward(self, x):
        f = self.emb_x(x.long())
        f.detach()

        gru_x, _ = self.rnn_gru_x(f)
        gru_x = self.dropout(gru_x)

        h_x = torch.mean(gru_x, 1)

        mu = self.aff_u_out(F.relu(self.aff_u_hid(h_x)))
        sigma = F.softplus(self.aff_s_out(F.relu(self.aff_s_hid(h_x))))

        return mu, sigma

class Encoder(nn.Module):
    def __init__(self, emb_x, config):
        super(Encoder, self).__init__()

        self.emb_x = emb_x

        self.aff_init_enc = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.rnn_bigru_x = nn.GRU(config["hidden_dim"], config["hidden_dim"], batch_first=True, bidirectional=True)

    def forward(self, x, z=None):
        batch_size = x.shape[0]

        f = self.emb_x(x.long())

        if z is not None:
            init_state = torch.unsqueeze(torch.tanh(self.aff_init_enc(z)), 0).expand(2, -1, -1).contiguous()

        enc_output, enc_hidden = self.rnn_bigru_x(f, init_state)

        fwd_enc_hidden = enc_hidden[0:enc_hidden.size(0):2]
        bwd_enc_hidden = enc_hidden[1:enc_hidden.size(0):2]
        enc_hidden = torch.cat([fwd_enc_hidden, bwd_enc_hidden], dim=2)
        return enc_output, enc_hidden

class Decoder(nn.Module):
    def __init__(self, vocab_tgt, emb_y, attention, config):
        super(Decoder, self).__init__()
        self.config = config
        self.device =  torch.device(config["device"])

        self.vocab_tgt = vocab_tgt
        self.vocab_size = len(vocab_tgt)
        self.sos_idx = vocab_tgt.stoi[config["sos"]]
        self.eos_idx = vocab_tgt.stoi[config["eos"]]
        self.pad_idx = vocab_tgt.stoi[config["pad"]]

        self.emb_y = emb_y
        self.attention = attention
        self.unk_idx = vocab_tgt.stoi[config["unk"]]

        self.dropout = nn.Dropout(config["dropout"])

        self.aff_init_dec = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.rnn_gru_dec = nn.GRU(config["emb_dim"] + 2 * config["hidden_dim"], config["hidden_dim"], batch_first=True)

        self.aff_out_y = nn.Linear(config["hidden_dim"] + config["emb_dim"], len(vocab_tgt))

    def forward_step(self, e_j, enc_output, x_mask, dec_hidden):
        query = dec_hidden.unsqueeze(1)
        c_j, _ = self.attention.forward(query, x_mask, enc_output)
        rnn_input = torch.cat((c_j, e_j), 2)
        dec_output, dec_hidden = self.rnn_gru_dec(rnn_input, dec_hidden.unsqueeze(0))
        pre_out = torch.cat((dec_hidden.squeeze(0).unsqueeze(1), e_j), 2)
        pre_out = self.dropout(pre_out)
        logits = self.aff_out_y(pre_out)
        return dec_output, dec_hidden.squeeze(0), logits


    def forward(self, enc_output, enc_hidden, x_mask, y=None, z=None, max_len=None):
        if max_len == None:
            max_len = y.shape[-1]

        # Init gru decoder with z
        if z is not None:
            dec_hidden = torch.tanh(self.aff_init_dec(z))

        self.attention.compute_proj_keys(enc_output)
        hidden_states = []
        logits_vectors = []
        for j in range(max_len):
            dec_input = y[:, j].unsqueeze(1).long()

            # Word dropout (excluded starting symbols) bugged
            # if j > 0:
            #     rw = Bernoulli(self.config["word_dropout"]).sample((dec_input.shape[0],))
            #     unk_idx = rw.nonzero().squeeze(1)
            #     dec_input[unk_idx, 0] = torch.tensor([self.unk_idx])


            e_j = self.emb_y(dec_input)
            dec_output, dec_hidden, logits = self.forward_step(e_j, enc_output, x_mask, dec_hidden)
            logits_vectors.append(logits)
        logits_vectors = torch.cat(logits_vectors, dim=1)
        return logits_vectors

    def predict(self, enc_output, enc_hidden, x_mask, z=None, n_best=1):
        size = self.config["beam_size"]
        batch_size = x_mask.shape[0]

        if z is not None:
            dec_hidden = torch.tanh(self.aff_init_dec(z))

        dec_hidden = tile(dec_hidden, size, dim=0)
        enc_output = tile(enc_output.contiguous(), size, dim=0)
        x_mask = tile(x_mask, size, dim=0)

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

        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (size - 1),
                                   device=self.device).repeat(
                                    batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["gold_score"] = [0] * batch_size

        for step in range(self.config["max_len"]):
            self.attention.compute_proj_keys(enc_output) # in loop, for shape

            dec_input = alive_seq[:, -1].view(-1, 1)
            e_j = self.emb_y(dec_input)
            dec_output, dec_hidden, logits = self.forward_step(e_j, enc_output, x_mask, dec_hidden)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

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

                dec_hidden = dec_hidden.index_select(0, select_indices)
                enc_output = enc_output.index_select(0, select_indices)
                x_mask = x_mask.index_select(0, select_indices)

        def pad_and_stack_hyps(hyps, pad_value):
            filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
            for j, h in enumerate(hyps):
                for k, i in enumerate(h):
                    filled[j, k] = i
            return filled

        final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=self.pad_idx)
        return final_outputs
