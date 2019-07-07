from configuration import setup_config
from joeynmt.vocabulary import Vocabulary

from torch.distributions.normal import Normal
import cond_nmt_utils as cond_nmt_utils
import aevnmt_utils as aevnmt_utils
import torch
from modules.search import greedy_lm, greedy
from modules.utils import tile_rnn_hidden

def create_model(vocab_src, vocab_tgt, config):
    if config["model_type"] == "cond_nmt":
        model = cond_nmt_utils.create_model(vocab_src, vocab_tgt, config)
    elif config["model_type"] == "aevnmt":
        model = aevnmt_utils.create_model(vocab_src, vocab_tgt, config)
    else:
        raise ValueError("Unknown model type: {}".format(config["model_type"]))
    print("Model: ", model)
    return model


if __name__ == '__main__':
    config = setup_config()

    # Get vocabularies
    src_vocab = config["data_dir"] + "/" + config["vocab_prefix"] + "." + config["src"]
    trg_vocab = config["data_dir"] + "/" + config["vocab_prefix"] + "." + config["tgt"]

    vocab_src = Vocabulary(file=src_vocab)
    vocab_tgt = Vocabulary(file=trg_vocab)

    model = create_model(vocab_src, vocab_tgt, config)
    model.to(torch.device(config["device"]))

    # Load checkpoint here
    checkpoints_path = "{}/{}".format(config["out_dir"], config["checkpoints_dir"])
    checkpoint = "aevnmt_run_luong-038"
    state = torch.load('{}/{}'.format(checkpoints_path, checkpoint))
    model.load_state_dict(state['state_dict'])

    with torch.no_grad():

        # Sample latent variable z
        batch_size = 5
        loc = torch.zeros([batch_size, config["latent_size"]])
        scale = torch.ones([batch_size, config["latent_size"]])
        qz = Normal(loc=loc, scale=scale)

        z = qz.sample().to(torch.device(config["device"]))

        lm_hidden = torch.tanh(model.lm_init_layer(z))
        lm_hidden = tile_rnn_hidden(lm_hidden, model.language_model.rnn)
        raw_lm_hypothesis = greedy_lm(
            model.language_model,
            model.emb_src,
            model.language_model.logits_layer,
            lm_hidden,
            vocab_src.stoi[config["sos"]],
            batch_size,
            config
        )

        x = torch.tensor(raw_lm_hypothesis).to(torch.device(config["device"]))
        x_mask = (x != vocab_src.stoi[config["pad"]]).unsqueeze(-2)
        enc_output, enc_hidden = model.encode(x, z)
        dec_hidden = model.init_decoder(enc_output, enc_hidden, z)

        raw_dec_hypothesis = greedy(model.decoder, model.emb_tgt,
            model.decoder.logits_layer, enc_output, dec_hidden, x_mask,
            vocab_tgt.stoi[config["sos"]], config)

        lm_hypotheses = vocab_src.arrays_to_sentences(raw_lm_hypothesis)
        dec_hypotheses = vocab_src.arrays_to_sentences(raw_dec_hypothesis)

        for i in range(len(lm_hypotheses)):
            print("Sentence ", i)
            print("x: ", ' '.join(lm_hypotheses[i]))
            print("y: ", ' '.join(dec_hypotheses[i]))
