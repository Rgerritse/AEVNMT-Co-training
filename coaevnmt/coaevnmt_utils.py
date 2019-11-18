from modules.inference import InferenceModel
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.language import LanguageModel
from modules.search import beam_search
from utils import create_attention, clean_sentences, compute_bleu, save_hypotheses, create_prev
from joeynmt import data
from joeynmt.batch import Batch
import torch
from tqdm import tqdm
from modules.search import ancestral_sample
from data_prep.constants import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from data_prep import batch_to_sentences, create_batch
from data_prep.utils import create_noisy_batch

def bi_train_fn(model_xy, model_yx, x_in, x_noisy_in, x_out, x_len, x_mask, y_in, y_noisy_in, y_out, y_len, y_mask, step):

    qz_xy = model_xy.inference(x_in, x_mask, x_len)
    qz_yx = model_yx.inference(y_in, y_mask, y_len)

    z_xy = qz_xy.rsample()
    z_yx = qz_yx.rsample()

    # Add z loss if works
    tm_logits_xy, lm_logits_xy, _, _ = model_xy(x_noisy_in, x_len, x_mask, y_noisy_in, z_xy)
    tm_logits_yx, lm_logits_yx, _, _ = model_yx(y_noisy_in, y_len, y_mask, x_noisy_in, z_yx)

    loss_xy = model_xy.loss(tm_logits_xy, lm_logits_xy, None, None, y_out, x_out, qz_xy, step)
    loss_yx = model_yx.loss(tm_logits_yx, lm_logits_yx, None, None, x_out, y_out, qz_yx, step)

    loss = loss_xy + loss_yx
    return loss

def mono_train_fn(model_xy, model_yx, y_in, y_noisy_in, y_len, y_mask, y_out, vocab_src, config, step):
    device = torch.device("cpu") if config["device"] == "cpu" else torch.device("cuda:0")
    with torch.no_grad():
        qz_y = model_yx.inference(y_in, y_mask, y_len)
        z_y = qz_y.sample()

        enc_output, enc_final = model_yx.encode(y_in, y_len, z_y)
        dec_hidden = model_yx.init_decoder(enc_output, enc_final, z_y)

        if config["decoding_method"] == "beam_search":
            x_samples = beam_search(model_yx.decoder,
                                    model_yx.emb_tgt,
                                    model_yx.generate_tm,
                                    enc_output,
                                    dec_hidden,
                                    y_mask,
                                    vocab_src.size(),
                                    vocab_src[SOS_TOKEN],
                                    vocab_src[EOS_TOKEN],
                                    vocab_src[PAD_TOKEN],
                                    config,
                                    beam_width=config["decoding_beam_width"],
                                    z=z_y)
        else:
            greedy = False if config["decoding_method"] == "ancestral" else True
            x_samples = ancestral_sample(model_yx.decoder,
                                         model_yx.emb_tgt,
                                         model_yx.generate_tm,
                                         enc_output,
                                         dec_hidden,
                                         y_mask,
                                         vocab_src[SOS_TOKEN],
                                         vocab_src[EOS_TOKEN],
                                         vocab_src[PAD_TOKEN],
                                         config,
                                         greedy=greedy,
                                         z=z_y)
        x_samples = batch_to_sentences(x_samples, vocab_src)
        x_in, x_out, x_mask, x_len, x_noisy_in = create_noisy_batch(
            x_samples, vocab_src, device, word_dropout=config["word_dropout"])
        x_mask = x_mask.unsqueeze(1)

    qz_x = model_xy.inference(x_in, x_mask, x_len)
    z_x = qz_x.rsample()
    tm_logits, lm_logits, _, _ = model_xy(x_noisy_in, x_len, x_mask, y_noisy_in, z_x)

    loss = model_xy.loss(tm_logits, lm_logits, None, None, y_out, x_out, qz_x, step)
    return loss
