from modules.inference import InferenceModel
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.language import LanguageModel
from models.coaevnmt import COAEVNMT
from modules.search import beam_search
from utils import create_attention, clean_sentences, compute_bleu, save_hypotheses, create_prev
from joeynmt import data
from joeynmt.batch import Batch
import torch
from tqdm import tqdm

def create_model(vocab_src, vocab_tgt, config):
    inference_model1 = InferenceModel(config)
    encoder1 = Encoder(config)
    attention1 = create_attention(config)
    decoder1 = Decoder(attention1, len(vocab_tgt), config)
    language_model1 = LanguageModel(len(vocab_src), config)

    inference_model2 = InferenceModel(config)
    encoder2 = Encoder(config)
    attention2 = create_attention(config)
    decoder2 = Decoder(attention2, len(vocab_tgt), config)
    language_model2 = LanguageModel(len(vocab_src), config)

    model = COAEVNMT(vocab_src, vocab_tgt, inference_model1, encoder1, decoder1,
        language_model1, inference_model2, encoder2, decoder2, language_model2,
        config)
    return model

def train_step(model, prev_x, x, x_mask, prev_y, y, y_mask,
    prev_y_mono, y_mono, y_mono_mask, prev_x_mono, x_mono, x_mono_mask, src_pad_idx, tgt_pad_idx, step):
    # Bilingual src2tgt model1
    qz1 = model.src_inference(prev_x, x_mask)
    z1 = qz1.rsample()
    tm1_logits, lm1_logits = model.forward_src2tgt(prev_x, x_mask, prev_y, z1)

    # Bilingual tgt2src model2
    qz2 = model.tgt_inference(prev_y, y_mask)
    z2 = qz2.rsample()
    tm2_logits, lm2_logits = model.forward_tgt2src(prev_y, y_mask, prev_x, z2)

    # Monolingual tgt
    sampled_x = model.sample_src(prev_y_mono, y_mono_mask)
    sampled_x = torch.from_numpy(sampled_x).to(x.device)
    sampled_x_mask = (sampled_x != src_pad_idx).unsqueeze(-2)

    qz3 = model.src_inference(sampled_x, sampled_x_mask)
    z3 = qz3.rsample()
    tm3_logits, lm3_logits = model.forward_src2tgt(sampled_x, sampled_x_mask, prev_y_mono, z3)

    # Monolingual src
    sampled_y = model.sample_tgt(prev_x_mono, x_mono_mask)
    sampled_y = torch.from_numpy(sampled_y).to(y.device)
    sampled_y_mask = (sampled_y != tgt_pad_idx).unsqueeze(-2)

    qz4 = model.tgt_inference(sampled_y, sampled_y_mask)
    z4 = qz4.rsample()
    tm4_logits, lm4_logits = model.forward_tgt2src(sampled_y, sampled_y_mask, prev_x_mono, z4)

    loss = model.loss(tm1_logits, lm1_logits, y, x, qz1,
        tm2_logits, lm2_logits, x, y, qz2,
        tm3_logits, lm3_logits, y_mono, sampled_x, qz3,
        tm4_logits, lm4_logits, x_mono, sampled_y, qz4, step)

    return loss

def validate(model, dataset_dev, vocab_src, vocab_tgt, epoch, config):
    model.eval()
    with torch.no_grad():
        model_hypotheses = []
        references = []

        dataloader = data.make_data_iter(dataset_dev, config["batch_size_eval"], train=False)
        for batch in tqdm(dataloader):
            cuda = False if config["device"] == "cpu" else True
            batch = Batch(batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)

            x = batch.src
            prev_x, x_mask = create_prev(x, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])
            y = batch.trg

            qz = model.src_inference(prev_x, x_mask)
            z = qz.mean

            enc_output, enc_hidden = model.encode_src(prev_x, z)
            dec_hidden = model.init_tgt_decoder(enc_output, enc_hidden, z)

            raw_hypothesis = beam_search(model.tgt_decoder, model.emb_tgt,
                model.tgt_decoder.logits_layer, enc_output, dec_hidden, x_mask, len(vocab_tgt),
                vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["eos"]],
                vocab_tgt.stoi[config["pad"]], config)

            model_hypotheses += vocab_tgt.arrays_to_sentences(raw_hypothesis)
            references += vocab_tgt.arrays_to_sentences(y)

        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        save_hypotheses(model_hypotheses, epoch, config)
        bleu = compute_bleu(model_hypotheses, references, epoch, config)
        return bleu
