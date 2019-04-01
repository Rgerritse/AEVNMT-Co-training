from fairseq.data import Dictionary, LanguagePairDataset
from models import AEVNMT
from tqdm import tqdm

def get_vocab(vocab_path):
    vocab = Dictionary()
    with open(vocab_path, encoding='utf-8') as f_vocab:
        for i, line in enumerate(f_vocab):
            vocab.add_symbol(line.strip())
    return vocab

def load_dataset(dataset, data_dir, src_lang, tgt_lang, vocab):
    print("Loading {} dataset".format(dataset))
    print("-- Reading source")
    src_path = "{}/{}.{}".format(data_dir, dataset, src_lang)
    src = []
    src_lengths = []
    with open(src_path, encoding='utf-8') as file:
        for line in tqdm(file):
        # for i, line in enumerate(file):
            sentence = "<s> " + line.strip()
            tokens = vocab.encode_line(sentence, add_if_not_exist=False)
            src.append(tokens)
            src_lengths.append(tokens.numel())

    print("-- Reading target")
    tgt_path = "{}/{}.{}".format(data_dir, dataset, tgt_lang)
    tgt = []
    tgt_lengths = []
    with open(tgt_path, encoding='utf-8') as file:
        for line in tqdm(file):
        # for i, line in enumerate(file):
            sentence = "<s> " + line.strip()
            tokens = vocab.encode_line(sentence, add_if_not_exist=False)
            tgt.append(tokens)
            tgt_lengths.append(tokens.numel())

    dataset = LanguagePairDataset(
            src=src,
            src_sizes=src_lengths,
            src_dict=vocab,
            tgt=tgt,
            tgt_sizes=src_lengths,
            tgt_dict=vocab,
        )
    return dataset

def setup_model(vocab, emb_dim, hidden_dim, max_len, device):
    sos_idx = vocab.index("<s>")
    eos_idx = vocab.index("</s>")
    model = AEVNMT(
        len(vocab),
        emb_dim,
        vocab.pad(),
        hidden_dim,
        max_len,
        device,
        train=True,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    ).to(device)
    return model
