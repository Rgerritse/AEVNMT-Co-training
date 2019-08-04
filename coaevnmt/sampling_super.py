import torch
from train_super import create_model
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
    print("x_embed: ", x_embed.shape)

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

def sample_from_posterior(model, batch, vocab_src, vocab_tgt, config):
    num_samples = 5

    x_out = batch.src
    x_out = x_out.repeat(num_samples, 1)

    x_in, x_mask = create_prev(x_out,  vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])


    qz = model.inference(x_in, x_mask)
    z = qz.sample()
    # prior = torch.distributions.Normal(loc=model.prior_loc, scale=model.prior_scale)
    # z = prior.sample(sample_shape=[num_samples])

    enc_output, enc_hidden = model.encode(x_in, z)
    # print(enc_output)
    # print("==")
    # print(enc_output[:,0,0])
    # asd
    # print(z)
    dec_hidden = model.init_decoder(enc_output, enc_hidden, z)
    # print("dec_hidden: ", dec_hidden[0].shape)
    # print("dec_hidden: ", dec_hidden)
    # asd
    # print(enc_output[:,0,0])
    # print(dec_hidden)

    # print(enc_output[:,0,0])
    # encoder_outputs, encoder_final = model.encode(x_in, z)
    #
    # # Create the initial hidden state of the TM.
    # hidden_tm = model.init_decoder(encoder_outputs, encoder_final, z)

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
                                 greedy=True)

    # print(y_samples)
    y_samples = vocab_tgt.arrays_to_sentences(y_samples)
    # pritn(y_samples)
    print("Sample translations from the approximate posterior")
    for idx, y in enumerate(y_samples, 1): print("{}: {}".format(idx, y))
    # y_samples = batch_to_sentences(y_samples, vocab_en)

    # Construct a post-processing pipeline for English.
    # postprocess = [Detokenizer("en"),
    #                Recaser("en"),
    #                WordDesegmenter(separator=hparams.subword_token)] # Executed in reverse order.
    # pipeline_en = Pipeline(pre=[], post=postprocess)
    #
    # # Print the samples.
    # pp_y_samples = [pipeline_en.post(y) for y in y_samples]
    # for idx, y in enumerate(pp_y_samples, 1): print(f"{idx}: {y}")

    # x_samples = ["in kleines blondes mädchen hält ein sandwich ."] * num_samples


def main():
    config = setup_config()
    config["train_prefix"] = 'sample'
    train_data, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    dataloader = data.make_data_iter(train_data, 1, train=True)
    sample = next(iter(dataloader))
    batch = Batch(sample, vocab_src.stoi[config["pad"]], use_cuda=False if config["device"] == "cpu" else True)

    model, _, validate_fn = create_model(vocab_src, vocab_tgt, config)
    model.to(torch.device(config["device"]))

    checkpoint_path = "output/aevnmt_word_dropout_0.1/checkpoints/aevnmt_word_dropout_0.1"
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])

    sample_from_latent(model, vocab_src, vocab_tgt, config)
    sample_from_posterior(model, batch, vocab_src, vocab_tgt, config)
    # Sample source sentences from the latent space

if __name__ == '__main__':
    main()
