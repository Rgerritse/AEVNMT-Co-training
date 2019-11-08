import torch
from tqdm import tqdm
from train_semi import create_models
from train_super import create_model
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

def printKL(model, dev_data, vocab_src, vocab_tgt, config, direction=None):
    model.eval()
    with torch.no_grad():
        model_hypotheses = []
        references = []

        device = torch.device("cpu") if config["device"] == "cpu" else torch.device("cuda:0")
        val_dl = DataLoader(dev_data, batch_size=config["batch_size_eval"],
                        shuffle=False, num_workers=4)
        # val_dl = BucketingParallelDataLoader(val_dl)
        # print(len(dev_data))
        kl = 0
        z_list = []
        for sentences_x, sentences_y in tqdm(val_dl):
            if direction == None or direction == "xy":
                sentences_x, sentences_y, sort_keys = sort_sentences(sentences_x, sentences_y)
                x_in, _, x_mask, x_len = create_batch(sentences_x, vocab_src, device)
                x_mask = x_mask.unsqueeze(1)
            else:
                sentences_y, sentences_x, sort_keys = sort_sentences(sentences_y, sentences_x)
                x_in, _, x_mask, x_len = create_batch(sentences_y, vocab_src, device)
                x_mask = x_mask.unsqueeze(1)

            qz = model.inference(x_in, x_mask, x_len)
            z = qz.mean
            z_list.append(z)

            enc_output, enc_hidden = model.encode(x_in, x_len, z)
            dec_hidden = model.init_decoder(enc_output, enc_hidden, z)

            pz = torch.distributions.Normal(loc=model.prior_loc, scale=model.prior_scale).expand(qz.mean.size())
            kl_loss = torch.distributions.kl.kl_divergence(qz, pz)
            kl_loss = kl_loss.sum(dim=1)
            kl += kl_loss.sum(dim=0)

            # kl_loss = kl_loss.sum(dim=0)

            # print(kl_loss.shape)
        kl /= len(dev_data)
        print("KL_loss: ", kl)
        z_tensor = torch.cat(z_list)
        print(z_tensor.mean(dim=0))
        print(z_tensor.var(dim=0))

# def getHistogram(model):


def main():
    # config = setup_config()
    config = setup_config()
    config["dev_prefix"] = "dev"
    # config["dev_prefix"] = "test_2016_flickr.lc.norm.tok"
    # config["dev_prefix"] = "test_2017_flickr.lc.norm.tok"

    vocab_src, vocab_tgt = load_vocabularies(config)
    _, dev_data, _ = load_data(config, vocab_src=vocab_src, vocab_tgt=vocab_tgt)

    checkpoint_path = "output/aevnmt_z_loss_en-de_run_1/checkpoints/aevnmt_z_loss_en-de_run_1"

    if config["model_type"] == "coaevnmt":
        model_xy, model_yx, _, _, validate_fn = create_models(vocab_src, vocab_tgt, config)
        model_xy.to(torch.device(config["device"]))
        model_yx.to(torch.device(config["device"]))

        state = torch.load(checkpoint_path)
        model_xy.load_state_dict(state['state_dict_xy'])
        model_yx.load_state_dict(state['state_dict_yx'])

        printKL(model_xy, dev_data, vocab_src, vocab_tgt, config, direction="xy")
        printKL(model_yx, dev_data, vocab_tgt, vocab_src, config, direction="yx")
    elif config["model_type"] == "aevnmt":
        model, _, _ = create_model(vocab_src, vocab_tgt, config)
        model.to(torch.device(config["device"]))

        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])

        printKL(model, dev_data, vocab_src, vocab_tgt, config, direction="None")

    # getHistogram(model_xy, dev_data, vocab_src, vocab_tgt, config, direction="xy")

if __name__ == '__main__':
    main()
