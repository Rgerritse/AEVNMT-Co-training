import argparse, json

def get_default_config():
    # Format: "option_name": (type, default_val, required, description)
    default_options = {
        # Data parameters
        "src": (str, None, False, "Source suffix, e.g., en"),
        "tgt": (str, None, False, "Target suffix, e.g., tr"),
        # "data_dir": (str, "data/en-tr", False, "Path to data directory"),
        # "train_prefix": (str, "bilingual/train_100000.en-tr", False, "Train prefix, expect files with src/tgt suffixes."), # Should be changed for real dataset
        "data_dir": (str, "data/multi30k", False, "Path to data directory"),
        "train_prefix": (str, "training", False, "Train prefix, expect files with src/tgt suffixes."), # Should be changed for real dataset
        "dev_prefix": (str, "dev", False, "Dev prefix, expect files with src/tgt suffixes."),
        "mono_prefix": (str, "comparable", False, "Monolingual files prefix, expect files with src/tgt suffixes."),
        "test_prefix": (str, "test", False, "Test prefix, expect files with src/tgt suffixes."),
        "back_prefix": (str, None, False, "Back-translation prefix, expect files with src/tgt suffixes."),
        "out_dir": (str, "output", False, "Path to output directory"),

        # Vocab
        # "vocab_prefix": (str, "aevnmt_vocab", False, "Vocab prefix, expect files with src/tgt suffixes."),
        "vocab_prefix": (str, "vocab_new", False, "Vocab prefix, expect files with src/tgt suffixes."),
        "sos": (str, "<s>", False, "Start-of-sentence symbol."),
        "eos": (str, "</s>", False, "End-of-sentence symbol."),
        "pad": (str, "<pad>", False, "Padding symbol."),
        "unk": (str, "<unk>", False, "Unknown symbol."),

        # Model
        "model_type": (str, None, False, "con_nmt|aevnmt|coaevnmt"),
        "emb_init_std": (float, 0.01, False, "Standard deviation of embeddings initialization"),
        "emb_size": (int, 512, False, "Dimensionality of word embeddings"),
        "hidden_size": (int, 512, False, "Dimensionality of hidden units"),
        "kl_free_nats": (float, 10.0, False, "Free bits value"),
        "kl_annealing_steps": (int, 0, False, "Number of steps to anneal kl loss"),
        "dropout": (float, 0.5, False, "Dropout"),
        "word_dropout": (float, 0.3, False, "Word Dropout"),
        "attention": (str, "bahdanau", False, "Attention type: bahdanau|luong"),
        "rnn_type": (str, "lstm", False, "Rnn type: gru|lstm"),
        "max_len": (int, 50, False, "Maximum sequence length"),
        "num_dec_layers": (int, 2, False, "Number of decoder RNN layers"),
        "num_enc_layers": (int, 1, False, "Number of encoder RNN layers"),
        "lr_reduce_cooldown": (int, 2, False, "Number of epochs to wait before resuming normal operation after lr has been reduced"),
        "lr_reduce_factor": (float, 0.5, False, "Factor by which the learning rate will be reduced"),
        "lr_reduce_patience": (int, 6, False, "Number of epochs with no improvement after which learning rate will be reduced"),
        "min_lr": (float, 5e-07, False, "Minimal learning rate"),
        "tied_embeddings": (bool, True, False, "Tie embeddings layer to output layer"),
        "pass_enc_final": (bool,  True, False, "Whether to pass encoder's hidden state to decoder when using an attention based model."),
        "share_vocab": (bool, False, False, "Whether to share vocabulary between source and target."),

        # Training
        "learning_rate": (float, 0.0005, False, "Learning rate"),
        "batch_size_train": (int, 64, False, "Number of samples per batch during training"),
        "max_gradient_norm": (float, 4.0, False, "Max norm of the gradients"),
        "latent_size": (int, 256, False, "Size of the latent variable"),
        "bilingual_warmup": (int, 10, False, "Number of epochs to train using only bilingual data"),

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
            option_type, _, _, _ = default_config[option]
            if option_type == bool and isinstance(value, str):
                value = value.lower() == "true"
            cmd_line_config[option] = value
    return cmd_line_config

def get_json_config(config_file):
    with open(config_file, 'r') as f:
        json_config = json.load(f)
    return json_config

def setup_config():
    config, default_options = get_default_config()

    cmd_line_config = get_cmd_line_config(default_options)

    if "config_file" in cmd_line_config:
        json_config = get_json_config(cmd_line_config["config_file"])
        config.update(json_config)
    config.update(cmd_line_config)

    print_config(config)
    return config

def print_config(config):
    print("Configuration: ")
    for i in sorted(config):
        print("  {}: {}".format(i, config[i]))
    print("\n")
