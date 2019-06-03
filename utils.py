import re
from joeynmt import data
from joeynmt.attention import BahdanauAttention, LuongAttention
from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
import subprocess
import sacrebleu
import torch

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

def load_mono_datasets(config, train_data):
    tok_fun = lambda s: s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=False,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=False,
                           include_lengths=True)

    mono_path = config["data_dir"] + "/" + config["mono_prefix"]
    train_src_mono = data.MonoDataset(mono_path, ".translation.en", src_field)
    train_tgt_mono = data.MonoDataset(mono_path, ".de", src_field)

    src_field.build_vocab(train_data)
    trg_field.build_vocab(train_data)
    return train_src_mono, train_tgt_mono

def create_prev_x(x, sos_idx, pad_idx):
    prev_x = []
    for t in range(x.shape[1]):
        if t == 0:
            prev_x.append(torch.empty(x[:, t:t+1].shape, dtype=torch.int64).fill_(sos_idx).to(x.device))
        else:
            prev_x.append(x[:, t-1:t])

    prev_x = torch.cat(prev_x, dim=1)
    prev_x_mask = (prev_x != pad_idx).unsqueeze(-2)
    return prev_x, prev_x_mask

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

def clean_sentences(hypotheses, references, config):
    subword_token = config["subword_token"]

    clean_hyps = []
    for sent in hypotheses:
        clean_sent = ' '.join(sent)
        clean_hyps.append(re.sub("({0} )|({0} ?$)|( {0})|(^ ?{0})".format(subword_token), "", clean_sent))

    clean_refs = []
    for sent in references:
        clean_sent = ' '.join(sent)
        clean_refs.append(re.sub("({0} )|({0} ?$)|( {0})|(^ ?{0})".format(subword_token), "", clean_sent))

    return clean_hyps, clean_refs


def save_hypotheses(hypotheses, epoch, config):
    file = '{}/{}/{}-{:03d}.{}'.format(config["out_dir"], config["predictions_dir"], config["session"], epoch, config["tgt"])
    with open(file, 'a') as the_file:
       for sent in hypotheses:
           the_file.write(sent + '\n')

def compute_bleu(hypotheses, references, epoch, config):
    bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references]).score
    file = '{}/{}-scores.txt'.format(config["out_dir"], config["session"])
    with open(file, 'a') as f_score:
        f_score.write("Epoch: {}, Bleu {}\n".format(epoch, bleu))
    return bleu




    # file_name = '{}/{}/{}-{:03d}.raw.{}'.format(config["out_dir"], config["predictions_dir"], config["session"], epoch, config["tgt"])
    # with open(file_name, 'a') as the_file:
    #    for sent in hypotheses:
    #        the_file.write(' '.join(sent) + '\n')
    # ref = "{}/{}.detok.{}".format(config["data_dir"], config["dev_prefix"], config["tgt"])
    # sacrebleu = subprocess.run(['./scripts/evaluate.sh',
    #     "{}/{}".format(config["out_dir"], config["predictions_dir"]),
    #     config["session"],
    #     '{:03d}'.format(epoch),
    #     ref,
    #     config["tgt"]],
    #     stdout=subprocess.PIPE)
    # bleu_score = sacrebleu.stdout.strip()
    # scores_file = '{}/{}-scores.txt'.format(config["out_dir"], config["session"])
    # with open(scores_file, 'a') as f_score:
    #     f_score.write("Epoch: {}, Bleu {}\n".format(epoch, bleu_score))
    # return bleu_score
