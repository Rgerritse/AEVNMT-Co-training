import torch
from tqdm import tqdm
from train_semi2 import create_models
from configuration import setup_config
from utils import load_dataset_joey, create_prev, clean_sentences, compute_bleu
from joeynmt import data
from joeynmt.batch import Batch
from modules.search import beam_search
import sacrebleu

def evaluate(model, dataset_dev, vocab_src, vocab_tgt, config, direction=None):
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
            # z = torch.zeros_like(z)

            enc_output, enc_hidden = model.encode(x_in, z)
            dec_hidden = model.init_decoder(enc_output, enc_hidden, z)

            raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                model.generate_tm, enc_output, dec_hidden, x_mask, len(vocab_tgt),
                vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["eos"]],
                vocab_tgt.stoi[config["pad"]], config)

            model_hypotheses += vocab_tgt.arrays_to_sentences(raw_hypothesis)
            references += vocab_tgt.arrays_to_sentences(y_out)

        model_hypotheses, references = clean_sentences(model_hypotheses, references, config)
        bleu = sacrebleu.raw_corpus_bleu(model_hypotheses, [references]).score
        print(bleu)

def main():
    config = setup_config()
    _, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    model_xy, model_yx, _, _, validate_fn = create_models(vocab_src, vocab_tgt, config)
    model_xy.to(torch.device(config["device"]))
    model_yx.to(torch.device(config["device"]))

    checkpoint_path = "output/coaevnmt_params/checkpoints/coaevnmt_params-046"
    state = torch.load(checkpoint_path)
    model_xy.load_state_dict(state['state_dict_xy'])
    model_yx.load_state_dict(state['state_dict_yx'])

    print("validation: {}-{}".format(config["src"], config["tgt"]))
    evaluate(model_xy, dev_data, vocab_src, vocab_tgt, config, direction="xy")

    print("validation: {}-{}".format(config["tgt"], config["src"]))
    evaluate(model_yx, dev_data, vocab_tgt, vocab_src, config, direction="yx")


if __name__ == '__main__':
    main()
