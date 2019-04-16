import argparse
from trainer import Trainer
from utils import get_vocabularies, load_dataset, setup_model
# from . import trainer

def add_arguments(parser):
    # Data
    parser.add_argument("--src", type=str, default="en", help="Source suffix, e.g., en")
    parser.add_argument("--tgt", type=str, default="de", help="Target suffix, e.g., tr") # Should be changed for real dataset
    parser.add_argument("--data_dir", type=str, default="data/multi30k", help="Path to data directory")
    parser.add_argument("--train_prefix", type=str, default="training", help="Train prefix, expect files with src/tgt suffixes.") # Should be changed for real dataset
    parser.add_argument("--dev_prefix", type=str, default="dev", help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default="test", help="Test prefix, expect files with src/tgt suffixes.")

    # Vocab
    parser.add_argument("--vocab_prefix", type=str, default="vocab", help="Vocab prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--sos", type=str, default="<s>", help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>", help="End-of-sentence symbol.")
    parser.add_argument("--pad", type=str, default="<pad>", help="Padding symbol.")
    parser.add_argument("--unk", type=str, default="<unk>", help="Unknown symbol.")

    # Model
    parser.add_argument("--model_type", type=str, default="nmt", help="nmt|aevnmt")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimensionality of word embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimensionality of hidden units")
    parser.add_argument("--dropout", type=int, default=0.3, help="Dropout")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length")

    # Training
    parser.add_argument("--learning_rate", type=int, default=0.0003, help="Learning rate")
    parser.add_argument("--batch_size_train", type=int, default=32, help="Number of samples per batch during training")
    parser.add_argument("--batch_size_eval", type=int, default=64, help="Number of samples per batch during evaluation")
    parser.add_argument("--num_steps", type=int, default=140000, help="Number of training steps")
    parser.add_argument("--steps_per_checkpoint", type=int, default=500, help="Number of steps per checkpoint")
    parser.add_argument("--steps_per_eval", type=int, default=500, help="Number of steps per eval")
    parser.add_argument("--kl_annealing_steps", type=int, default=80000, help="Number of steps for kl annealing")

    # Output
    parser.add_argument("--out_dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Name of directory where checkpoints are stored")
    parser.add_argument("--predictions_dir", type=str, default="predictions", help="Name of directory where predictions are stored")

    # Misc
    parser.add_argument("--session", type=str, default=None, required=True,  help="Name of sessions, used for output files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on: cuda|cpu")

def setup_config():
    return {
        # Data
        "src":FLAGS.src,
        "tgt":FLAGS.tgt,
        "data_dir":FLAGS.data_dir,
        "train_prefix":FLAGS.train_prefix,
        "dev_prefix":FLAGS.dev_prefix,
        "test_prefix":FLAGS.test_prefix,

        # Vocab
        "vocab_prefix":FLAGS.vocab_prefix,
        "sos":FLAGS.sos,
        "eos":FLAGS.eos,
        "pad":FLAGS.pad,
        "unk":FLAGS.unk,

        # Model
        "model_type":FLAGS.model_type,
        "emb_dim":FLAGS.emb_dim,
        "hidden_dim":FLAGS.hidden_dim,
        "max_len":FLAGS.max_len,
        "dropout":FLAGS.dropout,

        # Training
        "learning_rate":FLAGS.learning_rate,
        "batch_size_train":FLAGS.batch_size_train,
        "batch_size_eval":FLAGS.batch_size_eval,
        "num_steps":FLAGS.num_steps,
        "steps_per_checkpoint":FLAGS.steps_per_checkpoint,
        "steps_per_eval":FLAGS.steps_per_eval,
        "kl_annealing_steps":FLAGS.kl_annealing_steps,

        # Output
        "out_dir":FLAGS.out_dir,
        "checkpoints_dir":FLAGS.checkpoints_dir,
        "predictions_dir":FLAGS.predictions_dir,

        # Misc
        "session":FLAGS.session,
        "device":FLAGS.device
    }

def main():
    config = setup_config()
    vocab_src, vocab_tgt = get_vocabularies(config)
    dataset_train = load_dataset(config["train_prefix"], config, vocab_src, vocab_tgt)
    dataset_dev = load_dataset(config["dev_prefix"], config, vocab_src, vocab_tgt)
    model = setup_model(vocab_src, vocab_tgt, config)
    trainer = Trainer(model, vocab_src, vocab_tgt, dataset_train, dataset_dev, config)
    trainer.train_model()

if __name__ == '__main__':
    # REFER TO train_old.py
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    main()
