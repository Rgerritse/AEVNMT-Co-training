import torch
from tqdm import tqdm
from modules.inference import InferenceModel
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.language import LanguageModel
from models.aevnmt import AEVNMT
from modules.search import beam_search
from utils import create_attention, clean_sentences, compute_bleu, save_hypotheses, create_prev
from joeynmt import data
from joeynmt.batch import Batch

def create_model(vocab_src, vocab_tgt, config):
    inference_model = InferenceModel(config)
    encoder = Encoder(config)
    attention = create_attention(config)
    decoder = Decoder(attention, len(vocab_tgt), config)
    language_model = LanguageModel(len(vocab_src), config)
    model = AEVNMT(vocab_src, vocab_tgt, inference_model, encoder, decoder, language_model, config)
    return model

def train_step(model, x_in, x_noisy_in, x_out, x_mask, y_in, y_noisy_in, y_out, step):
    qz = model.inference(x_in, x_mask)
    z = qz.rsample()

    tm_logits, lm_logits = model(x_noisy_in, x_mask, y_in, z)
    loss = model.loss(tm_logits, lm_logits, y_out, x_out, qz, step)
    return loss

def validate(model, dataset_dev, vocab_src, vocab_tgt, epoch, config, direction=None):
    model.eval()
    with torch.no_grad():
        model_hypotheses = []
        references = []

        dataloader = data.make_data_iter(dataset_dev, config["batch_size_eval"], train=False)
        for batch in tqdm(dataloader):
            cuda = False if config["device"] == "cpu" else True
            batch = Batch(batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)

            if direction == None or direction == "xy":
                x_out = batch.src
                y_out = batch.trg
            elif direction == "yx":
                x_out = batch.trg
                y_out = batch.src
            x_in, x_mask = create_prev(x_out, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])

            qz = model.inference(x_in, x_mask)
            z = qz.mean

            enc_output, enc_hidden = model.encode(x_in, z)
            dec_hidden = model.init_decoder(enc_output, enc_hidden, z)

            raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                model.generate_tm, enc_output, dec_hidden, x_mask, len(vocab_tgt),
                vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["eos"]],
                vocab_tgt.stoi[config["pad"]], config)

            model_hypotheses += vocab_tgt.arrays_to_sentences(raw_hypothesis)
            references += vocab_tgt.arrays_to_sentences(y)

        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        save_hypotheses(model_hypotheses, epoch, config, direction)
        bleu = compute_bleu(model_hypotheses, references, epoch, config)
        return bleu
