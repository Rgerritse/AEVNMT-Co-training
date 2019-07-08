import json
import argparse
import cond_nmt_utils as cond_nmt_utils
import aevnmt_utils as aevnmt_utils

import torch
from trainer import Trainer
from utils import load_dataset_joey, create_attention
from modules.utils import init_model
from configuration import setup_config

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
    print("Model: ", model)
    return model, train_fn, validate_fn

def main():
    config = setup_config()

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
    main()
