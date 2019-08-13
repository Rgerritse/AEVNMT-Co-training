import sys
from configuration import setup_config
from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from data_prep import Vocabulary

def main():
    config = setup_config()

    en_files = [config["data_dir"] + "/" + "all_data." + config["src"]]
    de_files = [config["data_dir"] + "/" + "all_data." + config["tgt"]]

    vocab_src = Vocabulary().from_data(en_files, min_freq=0, max_size=sys.maxsize)
    vocab_tgt = Vocabulary().from_data(de_files, min_freq=0, max_size=sys.maxsize)

    # print(vocab_src)
    vocab_src.print_statistics()
    vocab_tgt.print_statistics()

    vocab_src.save(config["data_dir"] + "/" + "vocab_new." + config["src"])
    vocab_tgt.save(config["data_dir"] + "/" + "vocab_new." + config["tgt"])


if __name__ == '__main__':
    main()
