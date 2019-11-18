import torch
from tqdm import tqdm
from train_super import create_model
from configuration import setup_config
from utils import  load_vocabularies, load_data, create_prev, clean_sentences, compute_bleu, save_hypotheses
from joeynmt import data
from joeynmt.batch import Batch
from modules.search import beam_search
import sacrebleu
from torch.utils.data import DataLoader
from data_prep import BucketingParallelDataLoader, create_batch, batch_to_sentences
from data_prep.constants import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
import numpy as np

def main():
    config = setup_config()
    # config["dev_prefix"] = "dev"
    # config["dev_prefix"] = "comparable"
    # config["dev_prefix"] = "test_2016_flickr.lc.norm.tok"
    config["dev_prefix"] = "test_2017_flickr.lc.norm.tok"
    vocab_src, vocab_tgt = load_vocabularies(config)
    _, dev_data, _ = load_data(config, vocab_src=vocab_src, vocab_tgt=vocab_tgt)

    model, _, validate_fn = create_model(vocab_src, vocab_tgt, config)
    model.to(torch.device(config["device"]))

    # checkpoint_path = "output/cond_nmt_back_en-de_run_0/checkpoints/cond_nmt_back_en-de_run_0"
    # checkpoint_path = "output/cond_nmt_en-de_run_3/checkpoints/cond_nmt_en-de_run_3"
    # checkpoint_path = "output/cond_nmt_de-en_run_3/checkpoints/cond_nmt_de-en_run_3"
    # checkpoint_path = "output/cond_half_en-de_run_3/checkpoints/cond_half_en-de_run_3"
    # checkpoint_path = "output/cond_half_de-en_run_3/checkpoints/cond_half_de-en_run_3"
    # checkpoint_path = "output/cond_fourth_en-de_run_3/checkpoints/cond_fourth_en-de_run_3"
    # checkpoint_path = "output/cond_fourth_de-en_run_3/checkpoints/cond_fourth_de-en_run_3"
    # checkpoint_path = "output/cond_nmt_back_en-de_run_4/checkpoints/cond_nmt_back_en-de_run_4"


    # checkpoint_path = "output/aevnmt_final_fourth_en-de_run_3/checkpoints/aevnmt_final_fourth_en-de_run_3"
    # checkpoint_path = "output/aevnmt_final_fourth_de-en_run_3/checkpoints/aevnmt_final_fourth_de-en_run_3"

    # checkpoint_path = "output/aevnmt_final_half_en-de_run_3/checkpoints/aevnmt_final_half_en-de_run_3"
    # checkpoint_path = "output/aevnmt_final_half_de-en_run_3/checkpoints/aevnmt_final_half_de-en_run_3"
    # checkpoint_path = "output/aevnmt_z_to_pre_output_de-en_run_3/checkpoints/aevnmt_z_to_pre_output_de-en_run_3"

    # checkpoint_path = "output/cond_back_balanced_half_de-en_run_0/checkpoints/cond_back_balanced_half_de-en_run_0"
    # checkpoint_path = "output/cond_back_balanced_half_en-de_run_3/checkpoints/cond_back_balanced_half_en-de_run_3"


    # checkpoint_path = "output/cond_back_balanced_fourth_en-de_run_3/checkpoints/cond_back_balanced_fourth_en-de_run_3"
    checkpoint_path = "output/cond_back_balanced_fourth_de-en_run_3/checkpoints/cond_back_balanced_fourth_de-en_run_3"


    # checkpoint_path = "output/cond_back_balanced_de-en_run_3/checkpoints/cond_back_balanced_de-en_run_3"
    # checkpoint_path = "output/cond_back_balanced_fourth_en-de_run_0/checkpoints/cond_back_balanced_fourth_en-de_run_0"
    # checkpoint_path = "output/aevnmt_z_maxhid_en-de_run_0/checkpoints/aevnmt_z_maxhid_en-de_run_0"
    # checkpoint_path = "output/conmt_greedy_2en-de_run_0/checkpoints/conmt_greedy_2en-de_run_0"
    # checkpoint_path = "output/aevnmt_def_en-de_run_3/checkpoints/aevnmt_def_en-de_run_3"
    # checkpoint_path = "output/aevnmt_z_maxhid_de-en_run_3/checkpoints/aevnmt_z_maxhid_de-en_run_3"
    # checkpoint_path = "output/aevnmt_de-en_run_7/checkpoints/aevnmt_de-en_run_7"
    # checkpoint_path = "output/aevnmt_de-en_run_4/checkpoints/aevnmt_de-en_run_4"

    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])

    model.eval()
    device = torch.device("cpu") if config["device"] == "cpu" else torch.device("cuda:0")
    with torch.no_grad():
        model_hypotheses = []
        references = []
        original = []

        val_dl = DataLoader(dev_data, batch_size=config["batch_size_eval"],
                        shuffle=False, num_workers=4)
        # val_dl = BucketingParallelDataLoader(val_dl)
        for sentences_x, sentences_y in tqdm(val_dl):


            sentences_x = np.array(sentences_x)
            seq_len = np.array([len(s.split()) for s in sentences_x])
            sort_keys = np.argsort(-seq_len)
            sentences_x = sentences_x[sort_keys]
            # #
            sentences_y = np.array(sentences_y)

            x_in, _, x_mask, x_len = create_batch(sentences_x, vocab_src, device)
            x_mask = x_mask.unsqueeze(1)

            if config["model_type"] == "aevnmt":
                qz = model.inference(x_in, x_mask, x_len)
                z = qz.mean

                enc_output, enc_hidden = model.encode(x_in, x_len, z)
                dec_hidden = model.init_decoder(enc_output, enc_hidden, z)

                raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                    model.generate_tm, enc_output, dec_hidden, x_mask, vocab_tgt.size(),
                    vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                    vocab_tgt[PAD_TOKEN], config, z=z)
            else:
                enc_output, enc_hidden = model.encode(x_in, x_len)
                dec_hidden = model.decoder.initialize(enc_output, enc_hidden)

                raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                    model.generate_tm, enc_output, dec_hidden, x_mask, vocab_tgt.size(),
                    vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                    vocab_tgt[PAD_TOKEN], config)

            hypothesis = batch_to_sentences(raw_hypothesis, vocab_tgt)

            inverse_sort_keys = np.argsort(sort_keys)
            model_hypotheses += hypothesis[inverse_sort_keys].tolist()

            references += sentences_y.tolist()
            original += sentences_x[inverse_sort_keys].tolist()
        # save_hypotheses(model_hypotheses, 0, config, None)
        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        # print(model_hypotheses)
        # print(original)
        bleu = sacrebleu.raw_corpus_bleu(model_hypotheses, [references]).score
        print(bleu)

if __name__ == '__main__':
    main()
