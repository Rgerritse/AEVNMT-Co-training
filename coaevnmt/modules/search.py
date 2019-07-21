import torch
import torch.nn.functional as F
import numpy as np
from joeynmt.helpers import tile


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

def greedy(decoder, emb_tgt, logits_layer, enc_output, dec_hidden, x_mask, sos_idx, config):
    batch_size = x_mask.size(0)
    prev_y = x_mask.new_full(size=[batch_size, 1], fill_value=sos_idx,
                               dtype=torch.long)
    output = []
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

def beam_search(decoder, emb_tgt, logits_layer, enc_output, dec_hidden, x_mask, tgt_vocab_size, sos_idx, eos_idx, pad_idx, config):
    n_best = 1
    decoder.eval()
    with torch.no_grad():
        size = config["beam_width"]
        batch_size = x_mask.shape[0]

        if config["rnn_type"] == "gru":
            dec_hidden = tile(dec_hidden, size, dim=1)
        elif config["rnn_type"] == "lstm":
            h_n = tile(dec_hidden[0], size, dim=1)
            c_n = tile(dec_hidden[1], size, dim=1)
            dec_hidden = (h_n, c_n)

        enc_output = tile(enc_output.contiguous(), size, dim=0)
        x_mask = tile(x_mask, size, dim=0)
        decoder.attention.proj_keys = tile(decoder.attention.proj_keys, size, dim=0)

        batch_offset = torch.arange(batch_size, dtype=torch.long, device=enc_output.device)
        beam_offset = torch.arange(
            0,
            batch_size * size,
            step=size,
            dtype=torch.long,
            device=enc_output.device
        )
        alive_seq = torch.full(
            [batch_size * size, 1],
            sos_idx,
            dtype=torch.long,
            device=enc_output.device
        )

        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (size - 1),
                                   device=enc_output.device).repeat(
                                    batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["gold_score"] = [0] * batch_size

        for step in range(config["max_len"]):
            # decoder.attention.proj_keys = tile(decoder.attention.proj_keys, size, dim=0)
            # decoder.attention.compute_proj_keys(enc_output)
            prev_y = alive_seq[:, -1].view(-1, 1)
            embed_y = emb_tgt(prev_y)
            pre_output, dec_hidden = decoder.forward_step(embed_y, enc_output, x_mask, dec_hidden)
            logits = logits_layer(pre_output)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

            # multiply probs by the beam probability (=add logprobs)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            curr_scores = log_probs

            # compute length penalty
            if config["length_penalty"] > -1:
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** config["length_penalty"]
                curr_scores /= length_penalty

            # flatten log_probs into a list of possibilities
            curr_scores = curr_scores.reshape(-1, size * tgt_vocab_size)

            # pick currently best top k hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

            if config["length_penalty"] > -1:
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
            if step + 1 == config["max_len"]:
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

                decoder.attention.proj_keys = decoder.attention.proj_keys.index_select(0, select_indices)

                if config["rnn_type"] == "gru":
                    dec_hidden = dec_hidden.index_select(1, select_indices)
                elif config["rnn_type"] == "lstm":
                    h_n = dec_hidden[0].index_select(1, select_indices)
                    c_n = dec_hidden[1].index_select(1, select_indices)
                    dec_hidden = (h_n, c_n)

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
                                       pad_value=pad_idx)
        return final_outputs

# def beam_search(decoder, emb_tgt, generate_fn, enc_output, dec_hidden, x_mask, tgt_vocab_size, sos_idx, eos_idx, pad_idx, config):
#     n_best = 1
#     decoder.eval()
#     with torch.no_grad():
#         size = config["beam_width"]
#         batch_size = x_mask.shape[0]
#
#         if config["rnn_type"] == "gru":
#             dec_hidden = tile(dec_hidden, size, dim=1)
#         elif config["rnn_type"] == "lstm":
#             h_n = tile(dec_hidden[0], size, dim=1)
#             c_n = tile(dec_hidden[1], size, dim=1)
#             dec_hidden = (h_n, c_n)
#
#         enc_output = tile(enc_output.contiguous(), size, dim=0)
#         x_mask = tile(x_mask, size, dim=0)
#         decoder.attention.proj_keys = tile(decoder.attention.proj_keys, size, dim=0)
#
#         batch_offset = torch.arange(batch_size, dtype=torch.long, device=enc_output.device)
#         beam_offset = torch.arange(
#             0,
#             batch_size * size,
#             step=size,
#             dtype=torch.long,
#             device=enc_output.device
#         )
#         alive_seq = torch.full(
#             [batch_size * size, 1],
#             sos_idx,
#             dtype=torch.long,
#             device=enc_output.device
#         )
#
#         topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (size - 1),
#                                    device=enc_output.device).repeat(
#                                     batch_size))
#
#         # Structure that holds finished hypotheses.
#         hypotheses = [[] for _ in range(batch_size)]
#
#         results = {}
#         results["predictions"] = [[] for _ in range(batch_size)]
#         results["scores"] = [[] for _ in range(batch_size)]
#         results["gold_score"] = [0] * batch_size
#
#         for step in range(config["max_len"]):
#             prev_y = alive_seq[:, -1].view(-1, 1)
#             embed_y = emb_tgt(prev_y)
#             pre_output, dec_hidden = decoder.forward_step(embed_y, enc_output, x_mask, dec_hidden)
#             logits = generate_fn(pre_output)
#             log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
#
#             # multiply probs by the beam probability (=add logprobs)
#             log_probs += topk_log_probs.view(-1).unsqueeze(1)
#             curr_scores = log_probs
#
#             # compute length penalty
#             if config["length_penalty"] > -1:
#                 length_penalty = ((5.0 + (step + 1)) / 6.0) ** config["length_penalty"]
#                 curr_scores /= length_penalty
#
#             # flatten log_probs into a list of possibilities
#             curr_scores = curr_scores.reshape(-1, size * tgt_vocab_size)
#
#             # pick currently best top k hypotheses (flattened order)
#             topk_scores, topk_ids = curr_scores.topk(size, dim=-1)
#
#             if config["length_penalty"] > -1:
#                 # recover original log probs
#                 topk_log_probs = topk_scores * length_penalty
#
#              # reconstruct beam origin and true word ids from flattened order
#             topk_beam_index = topk_ids.div(tgt_vocab_size)
#             topk_ids = topk_ids.fmod(tgt_vocab_size)
#
#             # map beam_index to batch_index in the flat representation
#             batch_index = (
#                 topk_beam_index
#                 + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
#             select_indices = batch_index.view(-1)
#
#             # append latest prediction
#             alive_seq = torch.cat(
#                 [alive_seq.index_select(0, select_indices),
#                  topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len
#
#             is_finished = topk_ids.eq(eos_idx)
#             if step + 1 == config["max_len"]:
#                 is_finished.fill_(1)
#             # end condition is whether the top beam is finished
#             end_condition = is_finished[:, 0].eq(1)
#
#              # save finished hypotheses
#             if is_finished.any():
#                 predictions = alive_seq.view(-1, size, alive_seq.size(-1))
#                 for i in range(is_finished.size(0)):
#                     b = batch_offset[i]
#                     if end_condition[i]:
#                         is_finished[i].fill_(1)
#                     finished_hyp = is_finished[i].nonzero().view(-1)
#                     # store finished hypotheses for this batch
#                     for j in finished_hyp:
#                         hypotheses[b].append((
#                             topk_scores[i, j],
#                             predictions[i, j, 1:])  # ignore start_token
#                         )
#                     # if the batch reached the end, save the n_best hypotheses
#                     if end_condition[i]:
#                         best_hyp = sorted(
#                             hypotheses[b], key=lambda x: x[0], reverse=True)
#                         for n, (score, pred) in enumerate(best_hyp):
#                             if n >= n_best:
#                                 break
#                             results["scores"][b].append(score)
#                             results["predictions"][b].append(pred)
#                 non_finished = end_condition.eq(0).nonzero().view(-1)
#
#                  # if all sentences are translated, no need to go further
#                 # pylint: disable=len-as-condition
#                 if len(non_finished) == 0:
#                     break
#                 # remove finished batches for the next step
#                 topk_log_probs = topk_log_probs.index_select(0, non_finished)
#                 batch_index = batch_index.index_select(0, non_finished)
#                 batch_offset = batch_offset.index_select(0, non_finished)
#                 alive_seq = predictions.index_select(0, non_finished) \
#                     .view(-1, alive_seq.size(-1))
#
#                 # reorder indices, outputs and masks
#                 select_indices = batch_index.view(-1)
#
#                 decoder.attention.proj_keys = decoder.attention.proj_keys.index_select(0, select_indices)
#
#                 if config["rnn_type"] == "gru":
#                     dec_hidden = dec_hidden.index_select(1, select_indices)
#                 elif config["rnn_type"] == "lstm":
#                     h_n = dec_hidden[0].index_select(1, select_indices)
#                     c_n = dec_hidden[1].index_select(1, select_indices)
#                     dec_hidden = (h_n, c_n)
#
#                 enc_output = enc_output.index_select(0, select_indices)
#                 x_mask = x_mask.index_select(0, select_indices)
#
#         def pad_and_stack_hyps(hyps, pad_value):
#             filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
#                          dtype=int) * pad_value
#             for j, h in enumerate(hyps):
#                 for k, i in enumerate(h):
#                     filled[j, k] = i
#             return filled
#
#         final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
#                                         results["predictions"]],
#                                        pad_value=pad_idx)
#         return final_outputs
