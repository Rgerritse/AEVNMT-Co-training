from torch.utils.data import DataLoader
from fairseq.data import Dictionary, LanguagePairDataset
import argparse
from tqdm import tqdm

def add_arguments(parser):
    # Input paths
    parser.add_argument("--data_dir", type=str, default="data/setimes.tokenized.en-tr", help="Path to data directory")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="tr", help="Target language")
    parser.add_argument("--vocab", type=str, default="data/setimes.tokenized.en-tr/vocab.bpe", help="File to vocabulary")

def get_vocab():
    vocab = Dictionary()
    with open(FLAGS.vocab, encoding='utf-8') as f_vocab:
        for i, line in enumerate(f_vocab):
            vocab.add_symbol(line.strip())
    return vocab

def load_dataset(dataset, vocab):
    print("Loading {} dataset".format(dataset))

    print("Reading source")
    src_path = "{}/{}.{}".format(FLAGS.data_dir, dataset, FLAGS.src_lang)
    src = []
    src_lengths = []
    with open(src_path, encoding='utf-8') as file:
        # for line in tqdm(file):
        for i, line in enumerate(file):
            if i < 1000:
                sentence = line.strip()
                tokens = vocab.encode_line(sentence, add_if_not_exist=False)
                src.append(tokens)
                src_lengths.append(tokens.numel())

    print("Reading target")
    tgt_path = "{}/{}.{}".format(FLAGS.data_dir, dataset, FLAGS.tgt_lang)
    tgt = []
    tgt_lengths = []
    with open(tgt_path, encoding='utf-8') as file:
        # for line in tqdm(file):
        for i, line in enumerate(file):
            if i < 1000:
                sentence = line.strip()
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
            # left_pad_source=False
        )
    return dataset

def train(dataset):
    dataloader = DataLoader(dataset, 4, collate_fn=dataset.collater)
    for i, bla in enumerate(dataloader):
        print("")
        # if i == 0:
            # print(bla["target"])
            # print(bla["net_input"].keys())
        # print(i.shape)


def main():
    vocab = get_vocab()
    # print(vocab.pad())
    dataset_train = load_dataset("train", vocab)
    train(dataset_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    main()
