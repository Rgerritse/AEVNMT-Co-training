import torch
import sacrebleu
from tqdm import tqdm
from models.encoder import Encoder
from models.decoder import Decoder
from models.cond_nmt import CondNMT
from models.search import beam_search
from utils import create_attention, compute_bleu
from joeynmt import data
from joeynmt.batch import Batch


def create_model(vocab_src, vocab_tgt, config):
    encoder = Encoder(config)
    attention = create_attention(config)
    decoder = Decoder(attention, config)
    model = CondNMT(vocab_src, vocab_tgt, encoder, decoder, config)
    return model

def train_step(model, x, x_mask, prev, y):
    logits = model(x, x_mask, prev)
    loss = model.loss(logits, y, reduction="mean")
    return loss

def validate(model, dataset_dev, vocab_src, vocab_tgt, epoch, config):
    model.eval()
    with torch.no_grad():
        model_hypotheses = []

        dataloader = data.make_data_iter(dataset_dev, config["batch_size_eval"], train=False)
        for batch in tqdm(dataloader):
            cuda = False if config["device"] == "cpu" else True
            batch = Batch(batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)
            x = batch.src
            prev = batch.trg_input
            y = batch.trg

            x_mask = batch.src_mask
            prev_mask = batch.trg_mask

            enc_output, enc_hidden = model.encode(x)
            dec_hidden = model.decoder.initialize(enc_output, enc_hidden)

            raw_hypothesis = beam_search(model.decoder, model.emb_tgt,
                model.logits_layer, enc_output, dec_hidden, x_mask, len(vocab_tgt),
                vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["eos"]],
                vocab_tgt.stoi[config["pad"]], config)

            model_hypotheses += vocab_tgt.arrays_to_sentences(raw_hypothesis)

        bleu = compute_bleu(model_hypotheses, epoch, config)
        return bleu
             # with open(file_name, 'a') as the_file:
             #    for sent in model_hypotheses:
             #        the_file.write(' '.join(sent) + '\n')
             #
             # ref = "{}/{}.detok.{}".format(self.config["data_dir"], self.config["dev_prefix"], self.config["tgt"])
             #    sacrebleu = subprocess.run(['./scripts/evaluate.sh',
             #        "{}/{}".format(self.config["out_dir"], self.config["predictions_dir"]),
             #        self.config["session"],
             #        '{:03d}'.format(epoch),
             #        ref,
             #        self.config["tgt"]],
             #        stdout=subprocess.PIPE)
             #    bleu_score = sacrebleu.stdout.strip()
             #    scores_file = '{}/{}-scores.txt'.format(self.config["out_dir"], self.config["session"])
             #    with open(scores_file, 'a') as f_score:
             #        f_score.write("Epoch: {}, Bleu {}, Dev_loss: {}\n".format(epoch, bleu_score, total_loss))
