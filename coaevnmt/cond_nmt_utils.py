import torch
from tqdm import tqdm
from modules.encoder import Encoder
from modules.decoder import Decoder
from models.cond_nmt import CondNMT
from modules.search import beam_search
from utils import create_attention, clean_sentences, compute_bleu, save_hypotheses, create_prev
from joeynmt import data
from joeynmt.batch import Batch

def create_model(vocab_src, vocab_tgt, config):
    encoder = Encoder(config)
    attention = create_attention(config)
    decoder = Decoder(attention, len(vocab_tgt), config)
    model = CondNMT(vocab_src, vocab_tgt, encoder, decoder, config)
    return model

def train_step(model, x_in, x_noisy_in, x_out, x_len, x_mask, y_in, y_noisy_in, y_out, step):
    logits = model(x_noisy_in, x_mask, x_len, y_in)
    loss = model.loss(logits, y_out)
    return loss


# def train_step(model, x_in, x, x_mask, prev_y, y, step):
#     logits = model(x_in, x_mask, prev_y)
#     loss = model.loss(logits, y, reduction="mean")
#     return loss

# def validate(model, dataset_dev, vocab_src, vocab_tgt, epoch, config):
#     model.eval()
#     with torch.no_grad():
#         model_hypotheses = []
#         references = []
#
#         dataloader = data.make_data_iter(dataset_dev, config["batch_size_eval"], train=False)
#         for batch in tqdm(dataloader):
#             cuda = False if config["device"] == "cpu" else True
#             batch = Batch(batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)
#
#             x_out = batch.src
#             x_len = batch.src_lengths
#             x_in, x_mask = create_prev(x_out, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])
#             y_out = batch.trg
#
#             enc_output, enc_hidden = model.encode(x_in, x_len)
#             dec_hidden = model.decoder.initialize(enc_output, enc_hidden)
#
#             raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
#                 model.generate, enc_output, dec_hidden, x_mask, len(vocab_tgt),
#                 vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["eos"]],
#                 vocab_tgt.stoi[config["pad"]], config)
#
#             model_hypotheses += vocab_tgt.arrays_to_sentences(raw_hypothesis)
#             references += vocab_tgt.arrays_to_sentences(y_out)
#
#         model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
#         save_hypotheses(model_hypotheses, epoch, config)
#         bleu = compute_bleu(model_hypotheses, references, epoch, config)
#         return bleu

def validate(model, dataset_dev, vocab_src, vocab_tgt, epoch, config):
    model.eval()
    with torch.no_grad():
        model_hypotheses = []
        references = []

        dataloader = data.make_data_iter(dataset_dev, config["batch_size_eval"], train=False)
        for batch in tqdm(dataloader):
            cuda = False if config["device"] == "cpu" else True
            batch = Batch(batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)

            x_out = batch.src
            x_len = batch.src_lengths
            x_in, x_mask = create_prev(x_out, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])
            y_out = batch.trg




            enc_output, enc_hidden = model.encode(x_in)
            dec_hidden = model.decoder.initialize(enc_output, enc_hidden)

            raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                model.generate, enc_output, dec_hidden, x_mask, len(vocab_tgt),
                vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["eos"]],
                vocab_tgt.stoi[config["pad"]], config)

            model_hypotheses += vocab_tgt.arrays_to_sentences(raw_hypothesis)
            references += vocab_tgt.arrays_to_sentences(y_out)

        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        save_hypotheses(model_hypotheses, epoch, config)
        bleu = compute_bleu(model_hypotheses, references, epoch, config)
        return bleu
