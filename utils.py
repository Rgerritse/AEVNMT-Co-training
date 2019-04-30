# from fairseq.data import Dictionary, LanguagePairDataset
from joeynmt import vocabulary, data
# from models import AEVNMT
from models.baseline import Baseline
import torch
# from tqdm import tqdm

def get_vocabularies(config):
    print("Constructing vocabularies...")
    src_path = "{}/{}.{}".format(config["data_dir"], config["vocab_prefix"], config["src"])
    src_vocab = vocabulary.build_vocab(
        field="src",
        max_size=None,
        min_freq=None,
        dataset=None,
        vocab_file=src_path
    )

    tgt_path = "{}/{}.{}".format(config["data_dir"], config["vocab_prefix"], config["tgt"])
    tgt_vocab = vocabulary.build_vocab(
        field="tgt",
        max_size=None,
        min_freq=None,
        dataset=None,
        vocab_file=tgt_path
    )

    return src_vocab, tgt_vocab

def load_dataset_torchtext(config):
    src_field = data.Field(
        init_token=config["sos"],
        eos_token=config["eos"],
        batch_first=True,
        pad_token=config["pad"],
        unk_token=config["unk"],
        include_lengths=True
    )

    tgt_field = data.Field(
        init_token=None,
        eos_token=config["eos"],
        batch_first=True,
        pad_token=config["pad"],
        unk_token=config["unk"],
        include_lengths=True
    )

    src_path = "{}/{}.{}".format(config["data_dir"], data_prefix, config["src"])
    tgt_path = "{}/{}.{}".format(config["data_dir"], data_prefix, config["tgt"])

    train_data = TranslationDataset(
        path=train_path,
        exts=("." + config["src"], "." + config["tgt"]),
        fields=(src_field, tgt_field),
        filter_pred=
        lambda x: len(vars(x)['src'])
        <= max_sent_length
        and len(vars(x)['tgt'])
        <= max_sent_length
    )

    dev_data = TranslationDataset(
        path=dev_path,
        exts=("." + config["src"], "." + config["tgt"]),
        fields=(src_field, trg_field)
    )

    return train_data, dev_data

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
        "src_vocab": config["data_dir"] + "/" + config["vocab_prefix"] + "."+ config["src"],
        "trg_vocab": config["data_dir"] + "/" + config["vocab_prefix"] + "."+ config["tgt"]
    }
    train_data, dev_data, _, src_vocab, tgt_vocab = data.load_data(data_cfg)
    return train_data, dev_data, src_vocab, tgt_vocab

    # vocab_src = Dictionary(pad=config["pad"], eos=config["eos"], unk=config["unk"])
    # vocab_src.add_symbol(config["sos"])
    # src_path = "{}/{}.{}".format(config["data_dir"], config["vocab_prefix"], config["src"])
    # with open(src_path, encoding="utf-8") as f_vocab_src:
    #     for i, line in enumerate(f_vocab_src):
    #         vocab_src.add_symbol(line.strip())
    #
    #
    # vocab_tgt = Dictionary(pad=config["pad"], eos=config["eos"], unk=config["unk"])
    # vocab_tgt.add_symbol(config["sos"])
    # tgt_path = "{}/{}.{}".format(config["data_dir"], config["vocab_prefix"], config["tgt"])
    # with open(tgt_path) as f_vocab_src:
    #     for i, line in enumerate(f_vocab_src):
    #         vocab_tgt.add_symbol(line.strip())
    # return vocab_src, vocab_tgt


# def get_vocab(vocab_path):
#     vocab = Dictionary()
#     with open(vocab_path, encoding='utf-8') as f_vocab:
#         for i, line in enumerate(f_vocab):
#             vocab.add_symbol(line.strip())
#     return vocab
#
# def load_dataset(data_prefix, data_dir, src_lang, tgt_lang, vocab, num_sequences=None):
def load_dataset(data_prefix, config, vocab_src, vocab_tgt, shuffle, num_sequences=None):
    print("Loading {} dataset".format(data_prefix))

    print("-- Reading source")
    src_path = "{}/{}.{}".format(config["data_dir"], data_prefix, config["src"])
    src_seqs = []
    src_lengths = []
    with open(src_path, encoding='utf-8') as file:
        for i, line in enumerate(file):
            if config["num_seqs"] == None or i < config["num_seqs"]:
                sequence = "{} {}".format(config["sos"], line.strip())
                tokens = vocab_src.encode_line(sequence, add_if_not_exist=False)
                src_seqs.append(tokens)
                src_lengths.append(tokens.numel())

    # for seq in src_seqs:
    #     print(vocab_tgt.string(seq))

    print("-- Reading target")
    tgt_path = "{}/{}.{}".format(config["data_dir"], data_prefix, config["tgt"])
    tgt_seqs = []
    tgt_lengths = []
    with open(tgt_path, encoding='utf-8') as file:
        for i, line in enumerate(file):
            if config["num_seqs"] == None or i < config["num_seqs"]:
                sequence = "{} {}".format(line.strip(), config["eos"])
                tokens = vocab_tgt.encode_line(sequence, add_if_not_exist=False)
                tgt_seqs.append(tokens)
                tgt_lengths.append(tokens.numel())

    dataset = LanguagePairDataset(
            src=src_seqs,
            src_sizes=src_lengths,
            src_dict=vocab_src,
            tgt=tgt_seqs,
            tgt_sizes=src_lengths,
            tgt_dict=vocab_tgt,
            left_pad_source=False,
            shuffle=shuffle,
        )
    return dataset
#


    # data_cfg["src"] = config["src"]
def setup_model(vocab_src, vocab_tgt, config):
    device = torch.device(config["device"])
    if config["model_type"] == "nmt":
        model = Baseline(
            vocab_src,
            vocab_tgt,
            config
        ).to(device)
    else:
        model = AEVNMT(
            vocab_src,
            vocab_tgt,
            config
        ).to(device)
    return model
