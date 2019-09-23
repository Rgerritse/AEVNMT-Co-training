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

def bi_train_fn(model_xy, model_yx, x_in, x_noisy_in, x_out, x_len, x_mask, y_in, y_noisy_in, y_out, y_len, y_mask, step):
    logits_xy = model_xy(x_noisy_in, x_mask, x_len, y_noisy_in)
    logits_yx = model_yx(y_noisy_in, y_mask, y_len, x_noisy_in)

    loss_xy = model_xy.loss(logits_xy, y_out)
    loss_yx = model_yx.loss(logits_yx, x_out)

    loss = loss_xy + loss_yx
    return loss

def mono_train_fn(model_xy, model_yx, y_in, y_len, y_mask, y_out, vocab_src, config, step):
    device = torch.device("cpu") if config["device"] == "cpu" else torch.device("cuda:0")
    with torch.no_grad():
        enc_output, enc_final = model_yx.encode(y_in, y_len)
        dec_hidden = model_yx.init_decoder(enc_output, enc_final)

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
                                     greedy=config["greedy_sampling"])
        x_samples = batch_to_sentences(x_samples, vocab_src)
        x_in, x_out, x_mask, x_len = create_batch(x_samples, vocab_src, device)
        x_mask = x_mask.unsqueeze(1)
    logits = model_xy(x_in, x_mask, x_len, y_in)
    loss = model_xy.loss(logits, y_out)
    return loss
