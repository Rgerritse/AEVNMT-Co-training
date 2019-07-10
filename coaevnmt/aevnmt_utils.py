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

def train_step(model, prev_x, x, x_mask, prev_y, y, step):
    qz = model.inference(prev_x, x_mask)
    z = qz.rsample()

    tm_logits, lm_logits = model(prev_x, x_mask, prev_y, z)
    loss = model.loss(tm_logits, lm_logits, y, x, qz, step)
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

            if direction == None, direction == "xy":
                x = batch.src
                y = batch.trg
            elif direction == "yx":
                x = batch.trg
                y = batch.src
            prev_x, x_mask = create_prev(x, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])

            qz = model.inference(prev_x, x_mask)
            z = qz.mean

            enc_output, enc_hidden = model.encode(prev_x, z)
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
