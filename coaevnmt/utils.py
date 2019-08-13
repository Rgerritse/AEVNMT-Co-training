import os, re, sys
from joeynmt import data
from joeynmt.attention import BahdanauAttention, LuongAttention
from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from data_prep import Vocabulary, ParallelDataset, TextDataset
import subprocess
import sacrebleu
import torch

def load_vocabularies(config):
    vocab_src_file = config["data_dir"] + "/" + config["vocab_prefix"] + "."+ config["src"]
    vocab_tgt_file = config["data_dir"] + "/" + config["vocab_prefix"] + "."+ config["tgt"]

    vocab_src = Vocabulary.from_file(vocab_src_file, max_size=sys.maxsize)
    vocab_tgt = Vocabulary.from_file(vocab_tgt_file, max_size=sys.maxsize)
    return vocab_src, vocab_tgt

def load_data(config, vocab_src, vocab_tgt, use_memmap=False):
    train_src = config["data_dir"] + "/" + config["train_prefix"] + "." + config["src"]
    train_tgt = config["data_dir"] + "/" + config["train_prefix"] + "." + config["tgt"]
    val_src = config["data_dir"] + "/" + config["dev_prefix"] + "." + config["src"]
    val_tgt = config["data_dir"] + "/" + config["dev_prefix"] + "." + config["tgt"]
    opt_data = dict()

    training_data = ParallelDataset(train_src, train_tgt, max_length=config["max_len"])
    val_data = ParallelDataset(val_src, val_tgt, max_length=-1)

    if config["model_type"] == "coaevnmt":
        mono_src_path = "{}/{}.{}".format(config["data_dir"], config["mono_prefix"], config["src"])
        mono_tgt_path = "{}/{}.{}".format(config["data_dir"], config["mono_prefix"], config["tgt"])
        opt_data['mono_src'] = TextDataset(mono_src_path, max_length=config["max_len"])
        opt_data['mono_tgt'] = TextDataset(mono_tgt_path, max_length=config["max_len"])

    return training_data, val_data, opt_data

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

def load_mono_datasets(config, src_vocab, tgt_vocab):
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
    train_src_mono = data.MonoDataset(mono_path, ".{}".format(config["src"]), src_field)
    train_tgt_mono = data.MonoDataset(mono_path, ".{}".format(config["tgt"]), trg_field)

    src_field.vocab = src_vocab
    trg_field.vocab = tgt_vocab

    return train_src_mono, train_tgt_mono

def create_prev(x, sos_idx, pad_idx):
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

def create_optimizer(parameters, config):
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["lr_reduce_factor"],
        patience=config["lr_reduce_patience"],
        threshold=1e-2,
        threshold_mode="abs",
        cooldown=config["lr_reduce_cooldown"],
        min_lr=config["min_lr"]
    )
    return optimizer, scheduler

def clean_sentences(hypotheses, references, config):
    subword_token = config["subword_token"]

    clean_hyps = []
    for sent in hypotheses:
        # clean_sent = ' '.join(sent)
        clean_hyps.append(re.sub("({0} )|({0} ?$)|( {0})|(^ ?{0})".format(subword_token), "", sent))

    clean_refs = []
    for sent in references:
        # clean_sent = ' '.join(sent)
        clean_refs.append(re.sub("({0} )|({0} ?$)|( {0})|(^ ?{0})".format(subword_token), "", sent))

    return clean_hyps, clean_refs

def save_hypotheses(hypotheses, epoch, config, direction=None):
    hypotheses_path = '{}/{}/predictions'.format(config["out_dir"], config["session"])
    if not os.path.exists(hypotheses_path):
        os.makedirs(hypotheses_path)
    file = '{}/{}-{:03d}'.format(hypotheses_path, config["session"], epoch)
    if not direction is None:
        file += "-" + direction
    file += "." + config["tgt"]
    # print(hypotheses)
    with open(file, 'a') as the_file:
       for sent in hypotheses:
           # print(sent)
           the_file.write(sent + '\n')

def compute_bleu(hypotheses, references, epoch, config):
    bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references]).score
    # file = '{}/{}/{}-{:03d}.{}'.format(config["out_dir"], config["predictions_dir"], config["session"], epoch, config["tgt"])
    # ref = '{}/valid.detok.tr'.format(config["data_dir"])
    # process = subprocess.run(['./scripts/evaluate.sh', file, ref], stdout=subprocess.PIPE)
    # bleu = process.stdout.strip()
    scores = '{}/{}/bleu-scores.txt'.format(config["out_dir"], config["session"])
    with open(scores, 'a') as f_score:
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
