import torch
from fairseq.data import Dictionary, LanguagePairDataset
import argparse
from tqdm import tqdm
from models import AEVNMT, SentEmbInfModel
from trainer import Trainer
from utils import get_vocab, load_dataset, setup_model

def add_arguments(parser):
    # Paths
    parser.add_argument("--data_dir", type=str, default="data/setimes.tokenized.en-tr", help="Path to data directory")
    parser.add_argument("--vocab", type=str, default="data/setimes.tokenized.en-tr/vocab.bpe", help="File to vocabulary")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Path to checkpoints directory")
    parser.add_argument("--predictions_dir", type=str, default="predictions", help="Path to predictions directory")

    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="tr", help="Target language")

    # Network parameters
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimensionality of BPE embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimensionality of hidden units")

    # Training Parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train on cuda:0|cpu")
    parser.add_argument("--learning_rate", type=int, default=0.0003, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples per batch during training")
    parser.add_argument("--batch_size_eval", type=int, default=256, help="Number of samples per batch during evaluation")
    parser.add_argument("--beam_size", type=int, default=5, help="Number of candidate solutions for beam search")
    parser.add_argument("--model_name", type=str, default="AEVNMT", help="Name of the model (used for checkpoints)")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--num_sequences", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--num_steps", type=int, default=140000, help="Number of training steps")
    parser.add_argument("--steps_per_checkpoint", type=int, default=1, help="Number of steps per checkpoint")
    parser.add_argument("--steps_per_eval", type=int, default=1, help="Number of steps per eval")
    parser.add_argument("--kl_annealing_steps", type=int, default=80000, help="Number of steps for kl annealing")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab = get_vocab(FLAGS.vocab)
    dataset_train = load_dataset("train", FLAGS.data_dir, FLAGS.src_lang, FLAGS.tgt_lang, vocab, FLAGS.num_sequences)
    dataset_valid = load_dataset("valid", FLAGS.data_dir, FLAGS.src_lang, FLAGS.tgt_lang, vocab)
    model = setup_model(vocab, FLAGS.emb_dim, FLAGS.hidden_dim, FLAGS.max_len, device)
    trainer = Trainer(
        vocab,
        model,
        dataset_train,
        dataset_valid,
        FLAGS.model_name,
        FLAGS.num_steps,
        FLAGS.steps_per_checkpoint,
        FLAGS.steps_per_eval,
        FLAGS.kl_annealing_steps,
        FLAGS.device
    )
    trainer.run_epochs(FLAGS.learning_rate, vocab.pad(), len(vocab), FLAGS.batch_size, FLAGS.batch_size_eval, FLAGS.predictions_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    main()
