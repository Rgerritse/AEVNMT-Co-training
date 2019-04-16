from fairseq.data import Dictionary, LanguagePairDataset
from models import AEVNMT
import torch
# from tqdm import tqdm

def get_vocabularies(config):
    vocab_src = Dictionary(pad=config["pad"], eos=config["eos"], unk=config["unk"])
    src_path = "{}/{}.{}".format(config["data_dir"], config["vocab_prefix"], config["src"])
    with open(src_path, encoding="utf-8") as f_vocab_src:
        for i, line in enumerate(f_vocab_src):
            vocab_src.add_symbol(line.strip())

    vocab_tgt = Dictionary(pad=config["pad"], eos=config["eos"], unk=config["unk"])
    tgt_path = "{}/{}.{}".format(config["data_dir"], config["vocab_prefix"], config["tgt"])
    with open(tgt_path) as f_vocab_src:
        for i, line in enumerate(f_vocab_src):
            vocab_tgt.add_symbol(line.strip())
    return vocab_src, vocab_tgt


# def get_vocab(vocab_path):
#     vocab = Dictionary()
#     with open(vocab_path, encoding='utf-8') as f_vocab:
#         for i, line in enumerate(f_vocab):
#             vocab.add_symbol(line.strip())
#     return vocab
#
# def load_dataset(data_prefix, data_dir, src_lang, tgt_lang, vocab, num_sequences=None):
def load_dataset(data_prefix, config, vocab_src, vocab_tgt, num_sequences=None):
    print("Loading {} dataset".format(data_prefix))

    print("-- Reading source")
    src_path = "{}/{}.{}".format(config["data_dir"], data_prefix, config["src"])
    src_seqs = []
    src_lengths = []
    with open(src_path, encoding='utf-8') as file:
        for i, line in enumerate(file):
            if num_sequences == None or i < num_sequences:
                sequence = "{} {}".format(config["sos"], line.strip())
                tokens = vocab_src.encode_line(sequence, add_if_not_exist=False)
                src_seqs.append(tokens)
                src_lengths.append(tokens.numel())

    print("-- Reading target")
    tgt_path = "{}/{}.{}".format(config["data_dir"], data_prefix, config["tgt"])
    tgt_seqs = []
    tgt_lengths = []
    with open(tgt_path, encoding='utf-8') as file:
        for i, line in enumerate(file):
            if num_sequences == None or i < num_sequences:
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
            left_pad_source=False
        )
    return dataset

def setup_model(vocab_src, vocab_tgt, config):
    device = torch.device(config["device"])
    model = AEVNMT(
        vocab_src,
        vocab_tgt,
        config
    ).to(device)
    return model
