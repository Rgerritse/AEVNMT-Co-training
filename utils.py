from fairseq.data import Dictionary, LanguagePairDataset
from models import AEVNMT
from tqdm import tqdm

def get_vocab(vocab_path):
    vocab = Dictionary()
    with open(vocab_path, encoding='utf-8') as f_vocab:
        for i, line in enumerate(f_vocab):
            vocab.add_symbol(line.strip())
    return vocab

def load_dataset(dataset, data_dir, src_lang, tgt_lang, vocab, num_sequences=None):
    print("Loading {} dataset".format(dataset))
    print("-- Reading source")
    src_path = "{}/{}.{}".format(data_dir, dataset, src_lang)
    src = []
    src_lengths = []
    with open(src_path, encoding='utf-8') as file:
        for i, line in enumerate(file):
            if num_sequences == None or i < num_sequences:
                sentence = "<s> " + line.strip()
                tokens = vocab.encode_line(sentence, add_if_not_exist=False)
                src.append(tokens)
                src_lengths.append(tokens.numel())

    print("-- Reading target")
    tgt_path = "{}/{}.{}".format(data_dir, dataset, tgt_lang)
    tgt = []
    tgt_lengths = []
    with open(tgt_path, encoding='utf-8') as file:
        for i, line in enumerate(file):
            if num_sequences == None or i < num_sequences:
                sentence = line.strip() + " </s>"
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
            left_pad_source=False
        )
    return dataset

def setup_model(vocab, emb_dim, hidden_dim, max_len, device):
    sos_idx = vocab.index("<s>")
    eos_idx = vocab.index("</s>")
    pad_idx = vocab.pad()
    model = AEVNMT(
        vocab,
        len(vocab),
        emb_dim,
        vocab.pad(),
        hidden_dim,
        max_len,
        device,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx
    ).to(device)
    return model
