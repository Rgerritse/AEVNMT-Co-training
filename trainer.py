import os, torch, time, subprocess, subprocess
from joeynmt import data
from joeynmt.batch import Batch
from torch.utils.data import DataLoader
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
        # dataloader = DataLoader(self.dataset_train, self.config["batch_size_train"], collate_fn=self.dataset_train.collater)
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

        dataloader_iterator = iter(dataloader)
        for step in range(saved_step, self.config["num_steps"]):
            self.model.train()
            start_time = time.time()

            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                batch = next(dataloader_iterator)
            batch = Batch(batch, self.vocab_src.stoi[self.config["pad"]], use_cuda=True)

            # print(batch.src.tolist)
            # batch = batch[0]
            # print(batch)
            # batch = Batch(batch, self.pad_index, use_cuda=True)
            # asd
            opt.zero_grad()

            x = batch.src
            prev = batch.trg_input
            y = batch.trg

            x_mask = batch.src_mask
            prev_mask = batch.trg_mask
            # x_mask = (x != self.vocab_src.pad()).unsqueeze(-2)
            # prev_mask = (prev != self.vocab_src.pad())

            # x = batch["net_input"]["src_tokens"].to(self.device)
            # prev = batch["net_input"]["prev_output_tokens"].to(self.device)
            # y = batch["target"].to(self.device)
            #
            # x_mask = (x != self.vocab_src.pad()).unsqueeze(-2)
            # prev_mask = (prev != self.vocab_src.pad())

            pre_out_x, pre_out_y, mu_theta, sigma_theta = self.model.forward(x, x_mask, prev, prev_mask)
            loss, losses = self.compute_loss(pre_out_x, x, pre_out_y, y, mu_theta, sigma_theta, len(self.vocab_tgt), step + 1)
            loss.backward()
            opt.step()

            batch_spd = (time.time() - start_time)

            print("Step {:06d}/{:06d} , Total Loss: {:.2f}, Y-Loss: {:.2f}, Batch time: {:.1f}s".format(
                step + 1,
                self.config["num_steps"],
                loss.item(),
                losses[0].item(),
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
                    self.eval(step + 1)
                    # self.eval(self.model, self.dataset_valid, vocab_size, padding_idx, batch_size_eval, step + 1, predictions_dir)

    def eval(self, step):
        self.model.eval()
        print("Evaluating...")
        checkpoints_path = "{}/{}".format(self.config["out_dir"], self.config["predictions_dir"])
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        # dataloader = DataLoader(self.dataset_dev, self.config["batch_size_eval"], collate_fn=self.dataset_dev.collater)
        dataloader = data.make_data_iter(self.dataset_dev, self.config["batch_size_eval"], train=False)
        # dataloader = DataLoader(self.dataset_dev, self.config["batch_size_eval"], collate_fn=self.dataset_dev.collater)
        file_name = '{}/{}/{}-{:06d}.txt'.format(self.config["out_dir"], self.config["predictions_dir"], self.config["session"], step)
        total_loss = 0
        for batch in tqdm(iter(dataloader)):
            batch = Batch(batch, self.vocab_src.stoi[self.config["pad"]], use_cuda=True)
        # for batch in tqdm(dataloader):
            x = batch.src
            # prev = batch.trg_input
            # y = batch.trg

            x_mask = batch.src_mask
            # prev_mask = batch.trg_mask

            # x = batch["net_input"]["src_tokens"].to(self.device)
            # x_mask = (x != self.vocab_src.pad()).unsqueeze(-2)

            pred = self.model.predict(x, x_mask)
            # sort_idxs = np.argsort(batch["id"])
            # ordered_pred = torch.tensor([pred[i] for i in sort_idxs])
            # pre_out_x, pre_out_y, mu_theta, sigma_theta = self.model.forward(x, x_mask, y, y_mask)
            # loss, _ = self.compute_loss(pre_out_x, x, pre_out_y, y, mu_theta, sigma_theta, vocab_size, step + 1, reduction='sum')
            # total_loss += loss
            # total_loss = self.compute_loss(pre_out_y, y, mu_theta, sigma_theta, vocab_size, step, reduction='sum')
            # print(pred)
            # asd
            decoded = self.vocab_tgt.arrays_to_sentences(pred)
            with open(file_name, 'a') as the_file:
                for sent in decoded:
                    the_file.write(' '.join(sent) + '\n')
            # decoded = self.vocab_tgt.string(pred).replace(self.config["sos"], '').replace(self.config["eos"], '').replace(self.config["pad"], '').strip()
            # decoded = self.vocab_tgt.string(ordered_pred).replace(self.config["sos"], '').replace(self.config["eos"], '').replace(self.config["pad"], '').strip()


        # print("Validation Loss: {:.2f}".format(total_loss))

        # Remove BPE
        # output_file_name = '{}/{}-{:06d}-out.txt'.format(predictions_dir, self.model_name, step)
        # with open(output_file_name, "w") as file:
        #     sub = subprocess.run(['sed', '-r', 's/(@@ )|(@@ ?$)//g', file_name], stdout=file)
        #
        # with open(output_file_name) as inp, open(output_file_name + ".detok", "w") as out:
        #     subz = subprocess.run(['perl', 'data/mosesdecoder/scripts/tokenizer/detokenizer.perl', '-q'], stdin=inp, stdout=out)

        # val_path = "data/setimes.tokenized.en-tr/valid.tr"

        # sacrebleu = subprocess.run(['sacrebleu', '--input', output_file_name+".detok", self.valid_path, '--score-only'], stdout=subprocess.PIPE)
        valid_path = "{}/{}.{}".format(self.config["data_dir"], self.config["dev_prefix"], self.config["tgt"])
        sacrebleu = subprocess.run(['sacrebleu', '--input', file_name, valid_path, '--score-only'], stdout=subprocess.PIPE)
        # print(sacrebleu.stdout.strip())
        bleu_score = sacrebleu.stdout.strip()
        # bleu_score = sacrebleu.stdout.strip()
        scores_file = '{}/{}-scores.txt'.format(self.config["out_dir"], self.config["session"])
        with open(scores_file, 'a') as f_score:
            f_score.write("Step {}: {}\n".format(step, bleu_score))

    def compute_loss(self, pre_out_x, x, pre_out_y, y, mu, sigma, vocab_size, step, reduction='mean'):
        y_stack = torch.stack(pre_out_y, 1).view(-1, vocab_size)
        y_loss = F.cross_entropy(y_stack, y.long().view(-1), reduction=reduction)

        # x_stack = torch.stack(pre_out_x, 1).view(-1, vocab_size)
        # x_loss = F.cross_entropy(x_stack, x.long().view(-1), reduction=reduction)

        # KL_loss = self.compute_diagonal_gaussian_kl(mu, sigma)
        # if step < self.config["kl_annealing_steps"]:
        #     KL_loss *= step/self.config["kl_annealing_steps"]


        loss = y_loss
        losses = [y_loss]
        # loss = y_loss + x_loss + KL_loss
        # losses = [y_loss, x_loss, KL_loss]
        return loss, losses

    def compute_diagonal_gaussian_kl(self, mu, sigma):
        var = sigma ** 2
        loss = torch.mean(- 0.5 * torch.sum(torch.log(var) - mu ** 2 - var, 1))
        return loss
