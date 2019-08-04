import torch
from train_semi2 import create_models
from configuration import setup_config
from utils import load_dataset_joey, create_prev
from joeynmt import data
from joeynmt.batch import Batch
import sacrebleu
from torch.distributions.normal import Normal
from modules.utils import tile_rnn_hidden
from modules.search import ancestral_sample

def sample_from_latent(model, vocab_src, vocab_tgt, config):
    num_samples = 5

    prior = torch.distributions.Normal(loc=model.prior_loc, scale=model.prior_scale)
    z = prior.sample(sample_shape=[num_samples])

    hidden_lm = tile_rnn_hidden(model.lm_init_layer(z), model.language_model.rnn)
    x_init = z.new([vocab_tgt.stoi[config["sos"]] for _ in range(num_samples)]).long()
    x_embed = model.emb_src(x_init).unsqueeze(1)

    x_samples = [x_init.unsqueeze(-1)]

    for _ in range(config["max_len"]):
        pre_output, hidden_lm = model.language_model.forward_step(x_embed, hidden_lm)
        logits = model.generate_lm(pre_output)
        next_word_dist = torch.distributions.categorical.Categorical(logits=logits)
        x = next_word_dist.sample()
        x_embed = model.emb_src(x)
        x_samples.append(x)

    x_samples = torch.cat(x_samples, dim=-1)
    x_samples = vocab_src.arrays_to_sentences(x_samples)

    print("Sampled source sentences from the latent space ")
    for idx, x in enumerate(x_samples, 1): print("{}: {}".format(idx, x))

def sample_from_posterior(model, batch, vocab_src, vocab_tgt, config, direction="xy"):
    num_samples = 5

    if direction=="xy":
        x_out = batch.src
    else:
        x_out = batch.trg
    x_out = x_out.repeat(num_samples, 1)

    x_in, x_mask = create_prev(x_out,  vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])


    qz = model.inference(x_in, x_mask)
    z = qz.sample()

    enc_output, enc_hidden = model.encode(x_in, z)
    dec_hidden = model.init_decoder(enc_output, enc_hidden, z)

    # Sample target sentences conditional on the source and z.
    y_samples = ancestral_sample(model.decoder,
                                 model.emb_tgt,
                                 model.generate_tm,
                                 enc_output,
                                 dec_hidden,
                                 x_mask,
                                 vocab_tgt.stoi[config["sos"]],
                                 vocab_tgt.stoi[config["eos"]],
                                 vocab_tgt.stoi[config["pad"]],
                                 config,
                                 greedy=False)

    y_samples = vocab_tgt.arrays_to_sentences(y_samples)
    print("Sample translations from the approximate posterior")
    for idx, y in enumerate(y_samples, 1): print("{}: {}".format(idx, y))

def main():
    config = setup_config()
    config["train_prefix"] = 'sample'
    train_data, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    dataloader = data.make_data_iter(train_data, 1, train=True)
    sample = next(iter(dataloader))
    batch = Batch(sample, vocab_src.stoi[config["pad"]], use_cuda=False if config["device"] == "cpu" else True)

    model_xy, model_yx, _, _, validate_fn = create_models(vocab_src, vocab_tgt, config)
    model_xy.to(torch.device(config["device"]))
    model_yx.to(torch.device(config["device"]))

    checkpoint_path = "output/coaevnmt_params/checkpoints/coaevnmt_params-046"
    state = torch.load(checkpoint_path)
    model_xy.load_state_dict(state['state_dict_xy'])
    model_yx.load_state_dict(state['state_dict_yx'])


    # config = setup_config()
    # config["train_prefix"] = 'sample'
    # train_data, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    # dataloader = data.make_data_iter(train_data, 1, train=True)
    # sample = next(iter(dataloader))
    # batch = Batch(sample, vocab_src.stoi[config["pad"]], use_cuda=False if config["device"] == "cpu" else True)
    #
    # model, _, validate_fn = create_model(vocab_src, vocab_tgt, config)
    # model.to(torch.device(config["device"]))
    #
    # checkpoint_path = "output/aevnmt_word_dropout_0.1/checkpoints/aevnmt_word_dropout_0.1"
    # state = torch.load(checkpoint_path)
    # model.load_state_dict(state['state_dict'])

    print("validation: {}-{}".format(config["src"], config["tgt"]))
    sample_from_latent(model_xy, vocab_src, vocab_tgt, config)
    sample_from_posterior(model_xy, batch, vocab_src, vocab_tgt, config, direction="xy")
    print("")
    print("validation: {}-{}".format(config["tgt"], config["src"]))
    sample_from_latent(model_yx, vocab_tgt, vocab_src, config)
    sample_from_posterior(model_yx, batch, vocab_tgt, vocab_src, config, direction="yx")
    # Sample source sentences from the latent space

if __name__ == '__main__':
    main()
