import torch
from tqdm import tqdm
from train_semi import create_models
from configuration import setup_config
from utils import load_dataset_joey, create_prev, clean_sentences, compute_bleu
from joeynmt import data
from joeynmt.batch import Batch
from modules.search import beam_search
import sacrebleu
from utils import  load_vocabularies, load_data, create_prev, clean_sentences, compute_bleu
from data_prep.constants import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from torch.utils.data import DataLoader
from data_prep import BucketingParallelDataLoader, create_batch, batch_to_sentences
import numpy as np

def sort_sentences(sentences_src, sentences_tgt):
    sentences_src = np.array(sentences_src)
    seq_len = np.array([len(s.split()) for s in sentences_src])
    sort_keys = np.argsort(-seq_len)
    sentences_src = sentences_src[sort_keys]
    sentences_tgt = np.array(sentences_tgt)
    return sentences_src, sentences_tgt, sort_keys

def evaluate(model, dev_data, vocab_src, vocab_tgt, config, direction=None):
    model.eval()
    with torch.no_grad():
        model_hypotheses = []
        references = []

        device = torch.device("cpu") if config["device"] == "cpu" else torch.device("cuda:0")
        val_dl = DataLoader(dev_data, batch_size=config["batch_size_eval"],
                        shuffle=False, num_workers=4)
        # val_dl = BucketingParallelDataLoader(val_dl)
        for sentences_x, sentences_y in tqdm(val_dl):
            if direction == None or direction == "xy":
                sentences_x, sentences_y, sort_keys = sort_sentences(sentences_x, sentences_y)
                x_in, _, x_mask, x_len = create_batch(sentences_x, vocab_src, device)
                x_mask = x_mask.unsqueeze(1)
            else:
                sentences_y, sentences_x, sort_keys = sort_sentences(sentences_y, sentences_x)
                x_in, _, x_mask, x_len = create_batch(sentences_y, vocab_src, device)
                x_mask = x_mask.unsqueeze(1)

            if config["model_type"] == "coaevnmt":
                qz = model.inference(x_in, x_mask, x_len)
                z = qz.mean

                enc_output, enc_hidden = model.encode(x_in, x_len, z)
                dec_hidden = model.init_decoder(enc_output, enc_hidden, z)
            elif config["model_type"] == "conmt":
                enc_output, enc_hidden = model.encode(x_in, x_len)
                dec_hidden = model.decoder.initialize(enc_output, enc_hidden)

            raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                model.generate_tm, enc_output, dec_hidden, x_mask, vocab_tgt.size(),
                vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], config)

            hypothesis = batch_to_sentences(raw_hypothesis, vocab_tgt)
            inverse_sort_keys = np.argsort(sort_keys)
            model_hypotheses += hypothesis[inverse_sort_keys].tolist()

            if direction == None or direction == "xy":
                references += sentences_y.tolist()
            else:
                references += sentences_x.tolist()

        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        bleu = sacrebleu.raw_corpus_bleu(model_hypotheses, [references]).score
        print(bleu)

def main():
    # config = setup_config()
    config = setup_config()
    # config["dev_prefix"] = "dev"
    # config["dev_prefix"] = "test_2016_flickr.lc.norm.tok"
    config["dev_prefix"] = "test_2017_flickr.lc.norm.tok"

    vocab_src, vocab_tgt = load_vocabularies(config)
    _, dev_data, _ = load_data(config, vocab_src=vocab_src, vocab_tgt=vocab_tgt)
    model_xy, model_yx, _, _, validate_fn = create_models(vocab_src, vocab_tgt, config)
    model_xy.to(torch.device(config["device"]))
    model_yx.to(torch.device(config["device"]))

    # checkpoint_path = "output/coaevnmt_greedy_lm_off_run_5/checkpoints/coaevnmt_greedy_lm_off_run_5"
    # checkpoint_path = "output/coaevnmt_lr3_curriculum_en-de_run_4/checkpoints/coaevnmt_lr3_curriculum_en-de_run_4"
    # checkpoint_path = "output/coaevnmt_lr3_no_curriculum_no_warmup_en-de_run_4/checkpoints/coaevnmt_lr3_no_curriculum_no_warmup_en-de_run_4"
    # checkpoint_path = "output/coaevnmt_lr3_beam_dec_3_en-de_run_3/checkpoints/coaevnmt_lr3_beam_dec_3_en-de_run_3"
    # checkpoint_path = "output/conmt_anc_en-de_run_3/checkpoints/conmt_anc_en-de_run_3"
    # checkpoint_path = "output/conmt_greedy_en-de_run_3/checkpoints/conmt_greedy_en-de_run_3"
    # checkpoint_path = "output/conmt_beam_dec_3_2en-de_run_3/checkpoints/conmt_beam_dec_3_2en-de_run_3"
    # checkpoint_path = "output/conmt_beam_dec_5_2en-de_run_3/checkpoints/conmt_beam_dec_5_2en-de_run_3"
    checkpoint_path = "output/conmt_beam_dec_10_2en-de_run_3/checkpoints/conmt_beam_dec_10_2en-de_run_3"
    # checkpoint_path = "output/conmt_beam_dec_10_en-de_run_3/checkpoints/conmt_beam_dec_10_en-de_run_3"
    # checkpoint_path = "output/conmt_curc_diff_greedy_conv_yx_en-de_run_7/checkpoints/conmt_curc_diff_greedy_conv_yx_en-de_run_7"



    state = torch.load(checkpoint_path)
    model_xy.load_state_dict(state['state_dict_xy'])
    model_yx.load_state_dict(state['state_dict_yx'])

    print("validation: {}-{}".format(config["src"], config["tgt"]))
    evaluate(model_xy, dev_data, vocab_src, vocab_tgt, config, direction="xy")

    print("validation: {}-{}".format(config["tgt"], config["src"]))
    evaluate(model_yx, dev_data, vocab_tgt, vocab_src, config, direction="yx")


if __name__ == '__main__':
    main()
