import os, torch, time, subprocess, subprocess
from joeynmt import data
from joeynmt.batch import Batch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class Trainer():
    def __init__(self, model, vocab_src, vocab_tgt, dataset_train, dataset_dev, config):
        self.model = model
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.dataset_train = dataset_train
        self.dataset_dev = dataset_dev
        self.config = config
        self.device =  torch.device(config["device"])

    def train_model(self):
        print("Training...")
        checkpoints_path = "{}/{}".format(self.config["out_dir"], self.config["checkpoints_dir"])

        dataloader = data.make_data_iter(self.dataset_train, self.config["batch_size_train"], train=True)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        opt = torch.optim.Adam(parameters, lr=self.config["learning_rate"])

        saved_step = 0
        if os.path.exists(checkpoints_path):
            checkpoints = [cp for cp in sorted(os.listdir(checkpoints_path)) if '-'.join(cp.split('-')[:-1]) == self.config["session"]]
            if checkpoints:
                state = torch.load('{}/{}'.format(checkpoints_path, checkpoints[-1]))
                saved_step = state['step']
                self.model.load_state_dict(state['state_dict'])
                opt.load_state_dict(state['optimizer'])

        max_bleu = 0.0
        unimproved_bleu_checks = 0
        dataloader_iterator = iter(dataloader)
        for step in range(saved_step, self.config["num_steps"]):
            self.model.train()
            start_time = time.time()

            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                batch = next(dataloader_iterator)

            cuda = False if self.config["device"] == "cpu" else True
            batch = Batch(batch, self.vocab_src.stoi[self.config["pad"]], use_cuda=cuda)

            opt.zero_grad()

            x = batch.src
            prev = batch.trg_input
            y = batch.trg

            x_mask = batch.src_mask
            prev_mask = batch.trg_mask

            loss = self.model.forward(x, x_mask, prev, prev_mask, y, step + 1)

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config["max_gradient_norm"])
            opt.step()

            batch_spd = (time.time() - start_time)

            print("Step {:06d}/{:06d} , Loss: {:.2f}, Batch time: {:.1f}s".format(
                step + 1,
                self.config["num_steps"],
                loss.item(),
                batch_spd)
            )

            if (step + 1) % self.config["steps_per_checkpoint"] == 0:
                if not os.path.exists(checkpoints_path):
                    os.makedirs(checkpoints_path)
                state = {
                    'step': step + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': opt.state_dict(),
                }
                torch.save(state, '{}/{}-{:06d}'.format(checkpoints_path, self.config["session"], step + 1))

            if (step + 1) % self.config["steps_per_eval"] == 0:
                with torch.no_grad():
                    bleu_score = self.eval(step + 1)

                if float(bleu_score) > max_bleu:
                    max_bleu = float(bleu_score)
                    unimproved_bleu_checks = 0
                else:
                    unimproved_bleu_checks += 1
                    if unimproved_bleu_checks >= self.config["num_improv_checks"]:
                        break

    def eval(self, step):
        self.model.eval()
        print("Evaluating...")
        checkpoints_path = "{}/{}".format(self.config["out_dir"], self.config["predictions_dir"])
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        dataloader = data.make_data_iter(self.dataset_dev, self.config["batch_size_eval"], train=False)
        file_name = '{}/{}/{}-{:06d}.raw.{}'.format(self.config["out_dir"], self.config["predictions_dir"], self.config["session"], step, self.config["tgt"])
        total_loss = 0
        for batch in tqdm(iter(dataloader)):
            cuda = False if self.config["device"] == "cpu" else True
            batch = Batch(batch, self.vocab_src.stoi[self.config["pad"]], use_cuda=cuda)
            x = batch.src

            x_mask = batch.src_mask

            pred = self.model.predict(x, x_mask)
            decoded = self.vocab_tgt.arrays_to_sentences(pred)
            with open(file_name, 'a') as the_file:
                for sent in decoded:
                    the_file.write(' '.join(sent) + '\n')

        ref = "{}/{}.detok.{}".format(self.config["data_dir"], self.config["dev_prefix"], self.config["tgt"])
        sacrebleu = subprocess.run(['./scripts/evaluate.sh',
            "{}/{}".format(self.config["out_dir"], self.config["predictions_dir"]),
            self.config["session"],
            '{:06d}'.format(step),
            ref,
            self.config["tgt"]],
            stdout=subprocess.PIPE)
        bleu_score = sacrebleu.stdout.strip()
        scores_file = '{}/{}-scores.txt'.format(self.config["out_dir"], self.config["session"])
        with open(scores_file, 'a') as f_score:
            f_score.write("Step {}: {}\n".format(step, bleu_score))
        return bleu_score
