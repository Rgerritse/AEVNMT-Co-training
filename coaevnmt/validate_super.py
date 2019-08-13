import torch
from tqdm import tqdm
from train_super import create_model
from configuration import setup_config
from utils import  load_vocabularies, load_data, create_prev, clean_sentences, compute_bleu
from joeynmt import data
from joeynmt.batch import Batch
from modules.search import beam_search
import sacrebleu
from torch.utils.data import DataLoader
from data_prep import BucketingParallelDataLoader, create_batch, batch_to_sentences
from data_prep.constants import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

def main():
    config = setup_config()
    vocab_src, vocab_tgt = load_vocabularies(config)
    _, dev_data, _ = load_data(config, vocab_src=vocab_src, vocab_tgt=vocab_tgt)

    # _, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    model, _, validate_fn = create_model(vocab_src, vocab_tgt, config)
    model.to(torch.device(config["device"]))

    # checkpoint_path = "output/aevnmt_word_dropout_0.1/checkpoints/aevnmt_word_dropout_0.1"
    # checkpoint_path = "output/aevnmt_params_de-en/checkpoints/aevnmt_params_de-en"
    checkpoint_path = "output/aevnmt_new_kl_10_wd_0.1/checkpoints/aevnmt_new_kl_10_wd_0.1"
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])

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

            qz = model.inference(x_in, x_mask)
            z = qz.mean

            enc_output, enc_hidden = model.encode(x_in, z)
            dec_hidden = model.init_decoder(enc_output, enc_hidden, z)

            raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                model.generate_tm, enc_output, dec_hidden, x_mask, vocab_tgt.size(),
                vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], config)

            hypothesis = batch_to_sentences(raw_hypothesis, vocab_tgt)
            model_hypotheses += hypothesis.tolist()

            references += sentences_y.tolist()


        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        bleu = sacrebleu.raw_corpus_bleu(model_hypotheses, [references]).score
        print(bleu)

if __name__ == '__main__':
    main()
