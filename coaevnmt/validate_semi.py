import torch
from tqdm import tqdm
from train_semi2 import create_models
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

def evaluate(model, dev_data, vocab_src, vocab_tgt, config, direction=None):
    model.eval()
    with torch.no_grad():
        model_hypotheses = []
        references = []

        device = torch.device("cpu") if config["device"] == "cpu" else torch.device("cuda:0")
        val_dl = DataLoader(dev_data, batch_size=config["batch_size_eval"],
                        shuffle=False, num_workers=4)
        val_dl = BucketingParallelDataLoader(val_dl)
        for sentences_x, sentences_y in val_dl:
            if direction == None or direction == "xy":
                x_in, _, x_mask, x_len = create_batch(sentences_x, vocab_src, device)
                x_mask = x_mask.unsqueeze(1)
            else:
                x_in, _, x_mask, x_len = create_batch(sentences_y, vocab_src, device)
                x_mask = x_mask.unsqueeze(1)

        # dataloader = data.make_data_iter(dataset_dev, config["batch_size_eval"], train=False)
        # for batch in tqdm(dataloader):
        #     cuda = False if config["device"] == "cpu" else True
        #     batch = Batch(batch, vocab_tgt[EOS_TOKEN], use_cuda=cuda)
        #
        #     if direction == None or direction == "xy":
        #         x_out = batch.src
        #         y_out = batch.trg
        #     elif direction == "yx":
        #         x_out = batch.trg
        #         y_out = batch.src
        #     x_in, x_mask = create_prev(x_out, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])
            qz = model.inference(x_in, x_mask)
            z = qz.mean

            enc_output, enc_hidden = model.encode(x_in, z)
            dec_hidden = model.init_decoder(enc_output, enc_hidden, z)

            raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                model.generate_tm, enc_output, dec_hidden, x_mask, vocab_tgt.size(),
                vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], config)

            # qz = model.inference(x_in, x_mask)
            # z = qz.mean
            # # z = torch.zeros_like(z)
            #
            # enc_output, enc_hidden = model.encode(x_in, z)
            # dec_hidden = model.init_decoder(enc_output, enc_hidden, z)
            #
            # raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
            #     model.generate_tm, enc_output, dec_hidden, x_mask, len(vocab_tgt),
            #     vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
            #     vocab_tgt[PAD_TOKEN], config)

            hypothesis = batch_to_sentences(raw_hypothesis, vocab_tgt)
            model_hypotheses += hypothesis.tolist()

            if direction == None or direction == "xy":
                references += sentences_y.tolist()
            else:
                references += sentences_x.tolist()
            # references += sentences_y.tolist()


        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        bleu = sacrebleu.raw_corpus_bleu(model_hypotheses, [references]).score
        print(bleu)

def main():
    config = setup_config()
    config = setup_config()
    vocab_src, vocab_tgt = load_vocabularies(config)
    _, dev_data, _ = load_data(config, vocab_src=vocab_src, vocab_tgt=vocab_tgt)
    model_xy, model_yx, _, _, validate_fn = create_models(vocab_src, vocab_tgt, config)
    model_xy.to(torch.device(config["device"]))
    model_yx.to(torch.device(config["device"]))

    checkpoint_path = "output/coaevnmt_greedy_lm_off_run_5/checkpoints/coaevnmt_greedy_lm_off_run_5"
    state = torch.load(checkpoint_path)
    model_xy.load_state_dict(state['state_dict_xy'])
    model_yx.load_state_dict(state['state_dict_yx'])

    print("validation: {}-{}".format(config["src"], config["tgt"]))
    evaluate(model_xy, dev_data, vocab_src, vocab_tgt, config, direction="xy")

    print("validation: {}-{}".format(config["tgt"], config["src"]))
    evaluate(model_yx, dev_data, vocab_tgt, vocab_src, config, direction="yx")


if __name__ == '__main__':
    main()
