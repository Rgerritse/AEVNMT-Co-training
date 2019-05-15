from joeynmt import data
from joeynmt.attention import BahdanauAttention, LuongAttention
import subprocess

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

# Should be in model utils
def create_attention(config):
    key_size = 2 * config["hidden_size"]
    query_size = config["hidden_size"]

    if config["attention"] == "bahdanau":
        return BahdanauAttention(query_size, key_size, config["hidden_size"])
    elif config["attention"] == "luong":
        return LuongAttention(query_size, key_size)
    else:
        raise ValueError("Unknown attention: {}".format(config["attention"]))

def compute_bleu(hypotheses, epoch, config):
    file_name = '{}/{}/{}-{:03d}.raw.{}'.format(config["out_dir"], config["predictions_dir"], config["session"], epoch, config["tgt"])
    with open(file_name, 'a') as the_file:
       for sent in hypotheses:
           the_file.write(' '.join(sent) + '\n')
    ref = "{}/{}.detok.{}".format(config["data_dir"], config["dev_prefix"], config["tgt"])
    sacrebleu = subprocess.run(['./scripts/evaluate.sh',
        "{}/{}".format(config["out_dir"], config["predictions_dir"]),
        config["session"],
        '{:03d}'.format(epoch),
        ref,
        config["tgt"]],
        stdout=subprocess.PIPE)
    bleu_score = sacrebleu.stdout.strip()
    scores_file = '{}/{}-scores.txt'.format(config["out_dir"], config["session"])
    with open(scores_file, 'a') as f_score:
        f_score.write("Epoch: {}, Bleu {}\n".format(epoch, bleu_score))
    return bleu_score
