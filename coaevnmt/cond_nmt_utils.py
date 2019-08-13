import torch
from tqdm import tqdm
from modules.encoder import Encoder
from modules.decoder import Decoder
from models.cond_nmt import CondNMT
from modules.search import beam_search
from utils import create_attention, clean_sentences, compute_bleu, save_hypotheses, create_prev
from joeynmt import data
from joeynmt.batch import Batch
from data_prep import create_batch, batch_to_sentences, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from torch.utils.data import DataLoader
from data_prep import BucketingParallelDataLoader

def create_model(vocab_src, vocab_tgt, config):
    encoder = Encoder(config)
    attention = create_attention(config)
    decoder = Decoder(attention, vocab_tgt.size(), config)
    model = CondNMT(vocab_src, vocab_tgt, encoder, decoder, config)
    return model

def train_step(model, x_in, x_noisy_in, x_out, x_len, x_mask, y_in, y_noisy_in, y_out, step):
    logits = model(x_noisy_in, x_mask, x_len, y_noisy_in)
    loss = model.loss(logits, y_out)
    return loss

def validate(model, dev_data, vocab_src, vocab_tgt, epoch, config):
    model.eval()
    device = torch.device("cpu") if config["device"] == "cpu" else torch.device("cuda:0")
    with torch.no_grad():
        model_hypotheses = []
        references = []

        val_dl = DataLoader(dev_data, batch_size=config["batch_size_eval"],
                        shuffle=False, num_workers=4)
        val_dl = BucketingParallelDataLoader(val_dl)
        for sentences_x, sentences_y in val_dl:
            x_in, _, x_mask, x_len = create_batch(sentences_x, vocab_src, device)
            x_mask = x_mask.unsqueeze(1)

            enc_output, enc_hidden = model.encode(x_in)
            dec_hidden = model.decoder.initialize(enc_output, enc_hidden)

            raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                model.generate, enc_output, dec_hidden, x_mask, vocab_tgt.size(),
                vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], config)
            # raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
            #     model.generate, enc_output, dec_hidden, x_mask, len(vocab_tgt),
            #     vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["eos"]],
            #     vocab_tgt.stoi[config["pad"]], config)
            hypothesis = batch_to_sentences(raw_hypothesis, vocab_tgt)
            model_hypotheses += hypothesis.tolist()
            # model_hypotheses += vocab_tgt.arrays_to_sentences(raw_hypothesis)
            # model_hypotheses += vocab_tgt.arrays_to_sentences(raw_hypothesis)
            # references += vocab_tgt.arrays_to_sentences(y_out)
            references += sentences_y.tolist()
            # print(model_hypotheses)
            # asd

        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        save_hypotheses(model_hypotheses, epoch, config)
        bleu = compute_bleu(model_hypotheses, references, epoch, config)
        return bleu
