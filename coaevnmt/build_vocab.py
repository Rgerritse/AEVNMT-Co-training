from configuration import setup_config
from joeynmt import data

def load_dataset_joey(config):
    print("Creating datasets and vocabularies...")
    data_cfg = {
        "src": config["src"],
        "trg": config["tgt"],
        "train": config["data_dir"] + "/" + config["train_prefix"],
        "dev": config["data_dir"] + "/" + config["dev_prefix"],
        "level": "bpe",
        "lowercase": False,
        "max_sent_length": config["max_len"],
        "src_vocab": None,
        "trg_vocab": None
    }
    _, _, _, src_vocab, tgt_vocab = data.load_data(data_cfg)
    return src_vocab, tgt_vocab

def main():
    config = setup_config()
    vocab_src, vocab_tgt = load_dataset_joey(config)
    src_file = "{}/vocab.{}".format(config["data_dir"], config["src"])
    tgt_file = "{}/vocab.{}".format(config["data_dir"], config["tgt"])
    vocab_src.to_file(src_file)
    vocab_tgt.to_file(tgt_file)

if __name__ == '__main__':
    main()
