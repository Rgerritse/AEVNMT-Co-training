import torch
from torch.utils.data import DataLoader
from fairseq.data import Dictionary, LanguagePairDataset
import argparse
from tqdm import tqdm
from models import AEVNMT, SentEmbInfModel
import torch.nn.functional as F

def add_arguments(parser):
    # Input paths
    parser.add_argument("--data_dir", type=str, default="data/setimes.tokenized.en-tr", help="Path to data directory")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="tr", help="Target language")
    parser.add_argument("--vocab", type=str, default="data/setimes.tokenized.en-tr/vocab.bpe", help="File to vocabulary")

    # Network parameters
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimensionality of BPE embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimensionality of hidden units")

    # Training Parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train on cuda:0|cpu")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")

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
        for i, line in enumerate(file):
            sentence = "<s> " + line.strip()
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

def setup_model(vocab, device):
    s_tensor = torch.tensor([[vocab.index("<s>")]])
    model = AEVNMT(
        len(vocab),
        FLAGS.emb_dim,
        vocab.pad(),
        FLAGS.hidden_dim,
        device,
        train=True,
        s_tensor=s_tensor
    ).to(device)
    return model

def train(dataset, model, padding_idx, vocab_size, device):
    dataloader = DataLoader(dataset, 4, collate_fn=dataset.collater)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(parameters)

    # Add epoch loop
    for epoch in range(FLAGS.num_epochs):
        for i, batch in enumerate(dataloader):
            opt.zero_grad()
            x = batch["net_input"]["src_tokens"].to(device)
            y = batch["target"].to(device)

            x_mask = (x != padding_idx).unsqueeze(-2)
            y_mask = (y != padding_idx)

            pre_out_x, pre_out_y, mu_theta, sigma_theta = model.forward(x, x_mask, y, y_mask)
            loss = compute_loss(pre_out_x, pre_out_y, x, y, mu_theta, sigma_theta, vocab_size)
            loss.backward()
            opt.step()
            print("Epoch {}/{} , Loss: {}".format(epoch +1, FLAGS.num_epochs, loss.item()))

def compute_loss(pre_out_x, pre_out_y, x, y, mu_theta, sigma_theta, vocab_size):
    x_stack = torch.stack(pre_out_x, 1).view(-1, vocab_size)
    y_stack = torch.stack(pre_out_y, 1).view(-1, vocab_size)

    x_loss = F.cross_entropy(x_stack, x.long().view(-1))
    y_loss = F.cross_entropy(y_stack, y.long().view(-1))

    KL_loss =

    return x_loss + y_loss + KL_loss

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab = get_vocab()
    dataset_train = load_dataset("train", vocab)
    model = setup_model(vocab, device)
    train(dataset_train, model, vocab.pad(), len(vocab), device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    main()
