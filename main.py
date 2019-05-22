import json
import argparse
import cond_nmt_utils as cond_nmt_utils
import aevnmt_utils as aevnmt_utils

import torch
from trainer import Trainer
from utils import load_dataset_joey, create_attention
from modules.utils import init_model

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--src", type=str, default="en", help="Source suffix, e.g., en")
    parser.add_argument("--tgt", type=str, default="de", help="Target suffix, e.g., tr") # Should be changed for real dataset
    parser.add_argument("--data_dir", type=str, default="data/multi30k", help="Path to data directory")
    parser.add_argument("--train_prefix", type=str, default="training", help="Train prefix, expect files with src/tgt suffixes.") # Should be changed for real dataset
    parser.add_argument("--dev_prefix", type=str, default="dev", help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default="test", help="Test prefix, expect files with src/tgt suffixes.")

    # Vocab
    parser.add_argument("--vocab_prefix", type=str, default="vocab", help="Vocab prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--sos", type=str, default="<s>", help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>", help="End-of-sentence symbol.")
    parser.add_argument("--pad", type=str, default="<pad>", help="Padding symbol.")
    parser.add_argument("--unk", type=str, default="<unk>", help="Unknown symbol.")

    # Model
    parser.add_argument("--model_type", type=str, default="nmt", help="nmt|aevnmt")
    parser.add_argument("--emb_init_std", type=float, default=0.01, help="Standard deviation of embeddings initialization")
    parser.add_argument("--emb_size", type=int, default=256, help="Dimensionality of word embeddings")
    parser.add_argument("--hidden_size", type=int, default=256, help="Dimensionality of hidden units")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--word_dropout", type=float, default=0.3, help="Word Dropout")
    parser.add_argument("--attention", type=str, default="bahdanau", help="Attention type: bahdanau|luong")
    parser.add_argument("--rnn_type", type=str, default="gru", help="Rnn type: gru|lstm")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--pass_enc_final", type="bool", nargs="?", const=True, default=True, help="Whether to pass encoder's hidden state to decoder when using an attention based model.")
    parser.add_argument("--share_vocab", type="bool", nargs="?", const=True, default=False, help="Whether to share vocabulary between source and target.")

    # Training
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--batch_size_train", type=int, default=64, help="Number of samples per batch during training")
    parser.add_argument("--max_gradient_norm", type=float, default=4.0, help="Max norm of the gradients")
    parser.add_argument("--latent_size", type=int, default=32, help="Size of the latent variable")


    # Evaluation
    parser.add_argument("--batch_size_eval", type=int, default=64, help="Number of samples per batch during evaluation")
    parser.add_argument("--beam_width", type=int, default=10, help="Number of partial hypotheses")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Factor for length penalty")

    # Output
    parser.add_argument("--out_dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Name of directory where checkpoints are stored")
    parser.add_argument("--predictions_dir", type=str, default="predictions", help="Name of directory where predictions are stored")

    # Misc
    parser.add_argument("--session", type=str, default=None, required=True,  help="Name of sessions, used for output files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on: cuda|cpu")
    parser.add_argument("--patience", type=int, default=10, help="Number of checks whether metric-score has improved")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

def setup_config():
    config = {
        # Data
        "src":FLAGS.src,
        "tgt":FLAGS.tgt,
        "data_dir":FLAGS.data_dir,
        "train_prefix":FLAGS.train_prefix,
        "dev_prefix":FLAGS.dev_prefix,
        "test_prefix":FLAGS.test_prefix,

        # Vocab
        "vocab_prefix":FLAGS.vocab_prefix,
        "sos":FLAGS.sos,
        "eos":FLAGS.eos,
        "pad":FLAGS.pad,
        "unk":FLAGS.unk,

        # Model
        "model_type":FLAGS.model_type,
        "emb_init_std":FLAGS.emb_init_std,
        "emb_size":FLAGS.emb_size,
        "hidden_size":FLAGS.hidden_size,
        "max_len":FLAGS.max_len,
        "dropout":FLAGS.dropout,
        "word_dropout":FLAGS.word_dropout,
        "attention":FLAGS.attention,
        "pass_enc_final":FLAGS.pass_enc_final,
        "latent_size":FLAGS.latent_size,
        "share_vocab":FLAGS.share_vocab,

        # Training
        "learning_rate":FLAGS.learning_rate,
        "batch_size_train":FLAGS.batch_size_train,
        "max_gradient_norm":FLAGS.max_gradient_norm,

        # Evaluation
        "beam_width":FLAGS.beam_width,
        "batch_size_eval":FLAGS.batch_size_eval,
        # "steps_per_eval":FLAGS.steps_per_eval,
        "length_penalty":FLAGS.length_penalty,

        # Output
        "out_dir":FLAGS.out_dir,
        "checkpoints_dir":FLAGS.checkpoints_dir,
        "predictions_dir":FLAGS.predictions_dir,

        # Misc
        "session":FLAGS.session,
        "device":FLAGS.device,
        "patience":FLAGS.patience
    }

    # Loads config from json
    if FLAGS.config != None:
        with open(FLAGS.config, 'r') as f:
            config_json = json.load(f)
            for key, value in config_json.items():
                config[key] = value

    return config

def print_config(config):
    print("Configuration: ")
    for i in sorted(config):
        print("  {}: {}".format(i, config[i]))
    print("\n")

def create_model(vocab_src, vocab_tgt, config):
    if config["model_type"] == "cond_nmt":
        model = cond_nmt_utils.create_model(vocab_src, vocab_tgt, config)
        train_fn = cond_nmt_utils.train_step
        validate_fn = cond_nmt_utils.validate
    elif config["model_type"] == "aevnmt":
        model = aevnmt_utils.create_model(vocab_src, vocab_tgt, config)
        train_fn = aevnmt_utils.train_step
        validate_fn = aevnmt_utils.validate
    else:
        raise ValueError("Unknown model type: {}".format(config["model_type"]))
    print("model: ", model)
    return model, train_fn, validate_fn

def main():
    config = setup_config()
    print_config(config)
    train_data, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)

    model, train_fn, validate_fn = create_model(vocab_src, vocab_tgt, config)
    model.to(torch.device(config["device"]))

    init_model(
        model,
        vocab_src.stoi[config["pad"]],
        vocab_tgt.stoi[config["pad"]],
        config
    )

    trainer = Trainer(model, train_fn, validate_fn, vocab_src, vocab_tgt, train_data, dev_data, config)
    trainer.train_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    main()
