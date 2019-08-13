import torch
import torch.nn.functional as F
import numpy as np
# from joeynmt.helpers import tile
from torch.distributions.categorical import Categorical

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.
    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def greedy_lm(decoder, emb_tgt, logits_layer, lm_hidden, sos_idx, batch_size, config):
    prev_y = torch.full(size=[batch_size, 1], fill_value=sos_idx, dtype=torch.long,
                            device=lm_hidden[0].device)
    output = []
    for t in range(config["max_len"]):
        # decode one single step
        embed_y = emb_tgt(prev_y)
        pre_output, lm_hidden = decoder.forward_step(embed_y, lm_hidden)
        logits = logits_layer(pre_output)

        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(logits, dim=-1)  # batch x time=1
        output.append(next_word.squeeze(1).cpu().numpy())
        prev_y = next_word
        # attention_scores.append(att_probs.squeeze(1).cpu().numpy())
    stacked_output = np.stack(output, axis=1)  # batch, time
    return stacked_output


def ancestral_sample(decoder, emb_tgt, generate_fn, enc_output, dec_hidden, x_mask, sos_idx, eos_idx, pad_idx, config, greedy=False):
    batch_size = x_mask.size(0)
    prev_y = x_mask.new_full(size=[batch_size], fill_value=sos_idx,
                               dtype=torch.long)

    # print("prev_y: ", prev_y)
    predictions = []
    log_probs = []
    is_complete = torch.zeros_like(prev_y).unsqueeze(-1).byte()
    for t in range(config["max_len"]):
        embed_y = emb_tgt(prev_y)
        pre_output, dec_hidden = decoder.forward_step(embed_y, enc_output, x_mask, dec_hidden)
        logits = generate_fn(pre_output)
        py_x = Categorical(logits=logits)
        if greedy:
            prediction = torch.argmax(logits, dim=-1)
        else:
            prediction = py_x.sample()
        prev_y = prediction.view(batch_size)

        predictions.append(torch.where(is_complete, torch.full_like(prediction, pad_idx), prediction))
        is_complete = is_complete | (prediction == eos_idx)
    return torch.cat(predictions, dim=1)

def greedy(decoder, emb_tgt, logits_layer, enc_output, dec_hidden, x_mask, sos_idx, config):
    batch_size = x_mask.size(0)
    prev_y = x_mask.new_full(size=[batch_size, 1], fill_value=sos_idx,
                               dtype=torch.long)


    output = []
    is_complete = torch.zeros_like(prev_y).unsqueeze(-1).byte()
    for t in range(config["max_len"]):
        # decode one single step
        embed_y = emb_tgt(prev_y)
        pre_output, dec_hidden = decoder.forward_step(embed_y, enc_output, x_mask, dec_hidden)
        logits = logits_layer(pre_output)

        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(logits, dim=-1)  # batch x time=1
        output.append(next_word.squeeze(1).cpu().numpy())
        prev_y = next_word
        # attention_scores.append(att_probs.squeeze(1).cpu().numpy())
    stacked_output = np.stack(output, axis=1)  # batch, time
    return stacked_output

def beam_search(decoder, emb_tgt, generate, enc_output, dec_hidden, x_mask, tgt_vocab_size, sos_idx, eos_idx, pad_idx, config):
    """
    Beam search with size beam_width. Follows OpenNMT-py implementation.
    In each decoding step, find the k most likely partial hypotheses.
    :param decoder: an initialized decoder
    """
    n_best = 1
    decoder.eval()
    beam_width = config["beam_width"]
    alpha = config["length_penalty"]
    max_len = config["max_len"]
    with torch.no_grad():

        # Initialize the hjidden state and create the initial input.
        batch_size = x_mask.size(0)
        prev_y = torch.full(size=[batch_size], fill_value=sos_idx, dtype=torch.long,
                            device=x_mask.device)

        # Tile dec_hidden decoder states and encoder outputs beam_width times
        dec_hidden = tile(dec_hidden, beam_width, dim=1)    # [layers, B*beam_width, H_dec]
        decoder.attention.proj_keys = tile(decoder.attention.proj_keys,
                                           beam_width, dim=0)

        enc_output = tile(enc_output.contiguous(), beam_width,
                               dim=0)               # [B*beam_width, T_x, H_enc]
        x_mask = tile(x_mask, beam_width, dim=0)    # [B*beam_width, 1, T_x]

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=enc_output.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_width,
            step=beam_width,
            dtype=torch.long,
            device=enc_output.device)
        alive_seq = torch.full(
            [batch_size * beam_width, 1],
            sos_idx,
            dtype=torch.long,
            device=enc_output.device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_width - 1),
                                       device=enc_output.device).repeat(
                                        batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["gold_score"] = [0] * batch_size

        for step in range(max_len):
            prev_y = alive_seq[:, -1].view(-1)

            # expand current hypotheses, decode one single step
            prev_y = emb_tgt(prev_y)
            pre_output, dec_hidden = decoder.forward_step(prev_y, enc_output, x_mask, dec_hidden)
            logits = generate(pre_output)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)  # [B*beam_width, |V_y|]

            # multiply probs by the beam probability (=add logprobs)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            curr_scores = log_probs

            # compute length penalty
            if alpha > -1:
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
                curr_scores /= length_penalty

            # flatten log_probs into a list of possibilities
            curr_scores = curr_scores.reshape(-1, beam_width * tgt_vocab_size)

            # pick currently best top beam_width hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(beam_width, dim=-1)

            if alpha > -1:
                # recover original log probs
                topk_log_probs = topk_scores * length_penalty

            # reconstruct beam origin and true word ids from flattened order
            topk_beam_index = topk_ids.div(tgt_vocab_size)
            topk_ids = topk_ids.fmod(tgt_vocab_size)

            # map beam_index to batch_index in the flat representation
            batch_index = (
                topk_beam_index
                + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # append latest prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

            is_finished = topk_ids.eq(eos_idx)
            if step + 1 == max_len:
                is_finished.fill_(1)

            # end condition is whether the top beam is finished
            end_condition = is_finished[:, 0].eq(1)

            # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_width, alive_seq.size(-1))
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
            enc_output = enc_output.index_select(0, select_indices)
            x_mask = x_mask.index_select(0, select_indices)
            decoder.attention.proj_keys = decoder.attention.proj_keys. \
                    index_select(0, select_indices)

            if isinstance(dec_hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = dec_hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                dec_hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                dec_hidden = dec_hidden.index_select(1, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    # only works for n_best=1 for now
    assert n_best == 1

    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                        pad_value=pad_idx)

    return torch.from_numpy(final_outputs)
