import argparse, json

def get_default_config():
    # Format: "option_name": (type, default_val, required, description)
    default_options = {
        # Data parameters
        "src": (str, None, False, "Source suffix, e.g., en"),
        "tgt": (str, None, False, "Target suffix, e.g., tr"),
        "data_dir": (str, "data/en-tr", False, "Path to data directory"),
        "train_prefix": (str, "bilingual/train_100000.en-tr", False, "Train prefix, expect files with src/tgt suffixes."), # Should be changed for real dataset
        "dev_prefix": (str, "dev", False, "Dev prefix, expect files with src/tgt suffixes."),
        "test_prefix": (str, "test", False, "Test prefix, expect files with src/tgt suffixes."),
        "out_dir": (str, "output", False, "Path to output directory"),

        # Vocab
        "vocab_prefix": (str, "aevnmt_vocab", False, "Vocab prefix, expect files with src/tgt suffixes."),
        "sos": (str, "<s>", False, "Start-of-sentence symbol."),
        "eos": (str, "</s>", False, "End-of-sentence symbol."),
        "pad": (str, "<pad>", False, "Padding symbol."),
        "unk": (str, "<unk>", False, "Unknown symbol."),

        # Model
        "model_type": (str, "nmt", False, "nmt|aevnmt"),
        "emb_init_std": (float, 0.01, False, "Standard deviation of embeddings initialization"),
        "emb_size": (int, 512, False, "Dimensionality of word embeddings"),
        "hidden_size": (int, 512, False, "Dimensionality of hidden units"),
        "kl_free_nats": (float, 5.0, False, "Free bits value"),
        "dropout": (float, 0.3, False, "Dropout"),
        "word_dropout": (float, 0.3, False, "Word Dropout"),
        "attention": (str, "bahdanau", False, "Attention type: bahdanau|luong"),
        "rnn_type": (str, "gru", False, "Rnn type: gru|lstm"),
        "max_len": (int, 50, False, "Maximum sequence length"),
        "num_dec_layers": (int, 2, False, "Number of decoder RNN layers"),
        "num_enc_layers": (int, 1, False, "Number of encoder RNN layers"),
        "pass_enc_final": (bool,  True, False, "Whether to pass encoder's hidden state to decoder when using an attention based model."),
        "share_vocab": (bool, False, False, "Whether to share vocabulary between source and target."),

        # Training
        "learning_rate": (float, 0.0003, False, "Learning rate"),
        "batch_size_train": (int, 64, False, "Number of samples per batch during training"),
        "max_gradient_norm": (float, 4.0, False, "Max norm of the gradients"),
        "latent_size": (int, 32, False, "Size of the latent variable"),

        # Evaluation
        "batch_size_eval": (int, 64, False, "Number of samples per batch during evaluation"),
        "beam_width": (int, 10, False, "Number of partial hypotheses"),
        "length_penalty": (float, 1.0, False, "Factor for length penalty"),

        # Misc
        "session": (str, None, True,  "Name of sessions, used for output files"),
        "device": (str, "cuda", False, "Device to train on: cuda|cpu"),
        "patience": (int, 10, False, "Number of checks whether metric-score has improved"),
        "config_file": (str, None, False, "Path to config file")
    }

    config = {}
    for option in default_options.keys():
        _, default_value, _, _ = default_options[option]
        config[option] = default_value
    return config, default_options

def get_cmd_line_config(default_config):
    parser = argparse.ArgumentParser()
    for option in default_config.keys():
        option_type, _, required, description = default_config[option]
        option_type = str if option_type == bool else option_type
        parser.add_argument("--{}".format(option), type=option_type, required=required, help=description)
    flags = vars(parser.parse_known_args()[0])

    cmd_line_config = {}
    for option, value in flags.items():
        if not value is None:
            cmd_line_config[option] = value
    return cmd_line_config

def get_json_config(config_file):
    with open(config_file, 'r') as f:
        json_config = json.load(f)
    return json_config

def setup_config_new():
    config, default_options = get_default_config()

    cmd_line_config = get_cmd_line_config(default_options)

    if "config_file" in cmd_line_config:
        json_config = get_json_config(cmd_line_config["config_file"])
        config.update(json_config)
    config.update(cmd_line_config)
    
    print_config(config)
    return config

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--src", type=str, default="en", help="Source suffix, e.g., en")
    parser.add_argument("--tgt", type=str, default="tr", help="Target suffix, e.g., tr") # Should be changed for real dataset
    parser.add_argument("--data_dir", type=str, default="data/en-tr", help="Path to data directory")
    parser.add_argument("--train_prefix", type=str, default="bilingual/train_100000.en-tr", help="Train prefix, expect files with src/tgt suffixes.") # Should be changed for real dataset
    parser.add_argument("--dev_prefix", type=str, default="dev", help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default="test", help="Test prefix, expect files with src/tgt suffixes.")

    # Vocab
    parser.add_argument("--vocab_prefix", type=str, default="aevnmt_vocab", help="Vocab prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--sos", type=str, default="<s>", help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>", help="End-of-sentence symbol.")
    parser.add_argument("--pad", type=str, default="<pad>", help="Padding symbol.")
    parser.add_argument("--unk", type=str, default="<unk>", help="Unknown symbol.")

    # Model
    parser.add_argument("--model_type", type=str, default="nmt", help="nmt|aevnmt")
    parser.add_argument("--emb_init_std", type=float, default=0.01, help="Standard deviation of embeddings initialization")
    parser.add_argument("--emb_size", type=int, default=512, help="Dimensionality of word embeddings")
    parser.add_argument("--hidden_size", type=int, default=512, help="Dimensionality of hidden units")
    parser.add_argument("--kl_free_nats", type=float, default=5.0, help="Free bits value")
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

def print_config(config):
    print("Configuration: ")
    for i in sorted(config):
        print("  {}: {}".format(i, config[i]))
    print("\n")

def setup_config():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()

    config = {
        # Data
        "src":FLAGS.src,
        "tgt":FLAGS.tgt,
        "data_dir":FLAGS.data_dir,
        "train_prefix":FLAGS.train_prefix,
        "dev_prefix":FLAGS.dev_prefix,
        "test_prefix":FLAGS.test_prefix,
        "mono_prefix":"training.mono",
        # "src_mono_path":"training.mono.translation.en",
        # "tgt_mono_path":"training.mono.de",

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
        "kl_free_nats":FLAGS.kl_free_nats,
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

    print_config(config)
    return config
