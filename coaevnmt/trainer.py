import os, torch, time, subprocess, subprocess
from joeynmt import data
from joeynmt.batch import Batch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils import create_prev

class Trainer():
    def __init__(self, model, train_fn, validate_fn, vocab_src, vocab_tgt, dataset_train, dataset_dev, config):
        self.model = model
        self.train_fn = train_fn
        self.validate_fn = validate_fn
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.dataset_train = dataset_train
        self.dataset_dev = dataset_dev
        self.config = config
        self.device =  torch.device(config["device"])

        self.src_sos_idx = vocab_src.stoi[config["sos"]]
        self.src_eos_idx = vocab_src.stoi[config["eos"]]
        self.src_pad_idx = vocab_src.stoi[config["pad"]]
        self.src_unk_idx = vocab_src.stoi[config["unk"]]

        self.tgt_sos_idx = vocab_tgt.stoi[config["sos"]]
        self.tgt_eos_idx = vocab_tgt.stoi[config["eos"]]
        self.tgt_pad_idx = vocab_tgt.stoi[config["pad"]]
        self.tgt_unk_idx = vocab_tgt.stoi[config["unk"]]

    def train_model(self):
        print("Training...")
        checkpoints_path = "{}/checkpoints".format(self.config["out_dir"])

        dataloader = data.make_data_iter(self.dataset_train, self.config["batch_size_train"], train=True)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        opt = torch.optim.Adam(parameters, lr=self.config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=self.config["lr_reduce_factor"],
            patience=self.config["lr_reduce_patience"],
            threshold=1e-2,
            threshold_mode="abs",
            cooldown=self.config["lr_reduce_cooldown"],
            min_lr=self.config["min_lr"]
        )

        saved_epoch = 0
        patience_counter = 0
        max_bleu = 0.0
        if os.path.exists(checkpoints_path):
            checkpoints = [cp for cp in sorted(os.listdir(checkpoints_path)) if '-'.join(cp.split('-')[:-1]) == self.config["session"]]
            if checkpoints:
                state = torch.load('{}/{}'.format(checkpoints_path, checkpoints[-1]))
                saved_epoch = state['epoch']
                patience_counter = state['patience_counter']
                max_bleu = state['max_bleu']
                self.model.load_state_dict(state['state_dict'])
                opt.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])

        cuda = False if self.config["device"] == "cpu" else True
        for epoch in range(saved_epoch, self.config["num_epochs"]):
            for step, batch in enumerate(dataloader):
                self.model.train()

                batch = Batch(batch, self.vocab_src.stoi[self.config["pad"]], use_cuda=cuda)

                opt.zero_grad()

                x = batch.src
                prev_x, x_mask = create_prev(x, self.src_sos_idx, self.src_pad_idx)

                y = batch.trg
                prev_y = batch.trg_input

                if self.config["word_dropout"] > 0:
                    probs_prev_x = torch.zeros(prev_x.shape).uniform_(0, 1).to(prev_x.device)
                    prev_x = torch.where(
                        (probs_prev_x > self.config["word_dropout"]) | (prev_x == self.src_pad_idx) | (prev_x == self.src_eos_idx),
                        prev_x,
                        torch.empty(prev_x.shape, dtype=torch.int64).fill_(self.src_unk_idx).to(prev_x.device)
                    )
                    probs_prev_y = torch.zeros(prev_y.shape).uniform_(0, 1).to(prev_y.device)
                    prev_y = torch.where(
                        (probs_prev_y > self.config["word_dropout"]) | (prev_y == self.tgt_pad_idx) | (prev_y == self.tgt_eos_idx),
                        prev_y,
                        torch.empty(prev_y.shape, dtype=torch.int64).fill_(self.tgt_unk_idx).to(prev_y.device)
                    )

                loss = self.train_fn(self.model, prev_x, x, x_mask, prev_y, y, step)
                loss.backward()

                if self.config["max_gradient_norm"] > 0:
                    clip_grad_norm_(self.model.parameters(), self.config["max_gradient_norm"])

                opt.step()

                print("Epoch: {:03d}/{:03d}, Batch {:05d}/{:05d}, Loss: {:.2f}".format(
                    epoch + 1,
                    self.config["num_epochs"],
                    step + 1,
                    len(dataloader),
                    loss.item())
                )

                val_bleu = self.evaluate(epoch)
            scheduler.step(float(val_bleu))

            # Save checkpoint
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            state = {
                'epoch': epoch + 1,
                'patience_counter': patience_counter,
                'max_bleu': max_bleu,
                'state_dict': self.model.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, '{}/{}-{:03d}'.format(checkpoints_path, self.config["session"], epoch + 1))

            print("Blue score: {}".format(val_bleu))
            if float(val_bleu) > max_bleu:
                max_bleu = float(val_bleu)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    break

    def evaluate(self, epoch):
        checkpoints_path = "{}/checkpoints".format(self.config["out_dir"])
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        self.model.eval()
        val_bleu = self.validate_fn(
            self.model,
            self.dataset_dev,
            self.vocab_src,
            self.vocab_tgt,
            epoch + 1,
            self.config
        )
        return val_bleu
