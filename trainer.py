import os
import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
import subprocess
from tqdm import tqdm
import os

class Trainer():
    def __init__(
            self,
            vocab,
            model,
            dataset_train,
            dataset_valid,
            valid_path,
            model_name,
            num_steps,
            steps_per_checkpoint,
            steps_per_eval,
            kl_annealing_steps,
            device,
            checkpoints_dir="checkpoints",
            predictions_dir="predictions",
        ):
        self.vocab = vocab
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.valid_path = valid_path
        self.model_name = model_name
        self.num_steps = num_steps
        self.steps_per_checkpoint = steps_per_checkpoint
        self.steps_per_eval = steps_per_eval
        self.kl_annealing_steps = kl_annealing_steps
        self.checkpoints_dir = checkpoints_dir
        self.predictions_dir = predictions_dir
        self.device = device

    def run_epochs(self, learning_rate, padding_idx, vocab_size, batch_size, batch_size_eval, predictions_dir):
        dataloader = DataLoader(self.dataset_train, batch_size, collate_fn=self.dataset_train.collater)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        opt = torch.optim.Adam(parameters, lr=learning_rate)

        saved_step = 0
        if os.path.exists(self.checkpoints_dir):
            checkpoints = [cp for cp in sorted(os.listdir('checkpoints')) if '-'.join(cp.split('-')[:-1]) == self.model_name]
            if checkpoints:
                state = torch.load('checkpoints/{}'.format(checkpoints[-1]))
                saved_step = state['step']
                self.model.load_state_dict(state['state_dict'])
                opt.load_state_dict(state['optimizer'])

        num_batches = len(dataloader)
        dataloader_iterator = iter(dataloader)
        for step in range(saved_step, self.num_steps):
            start_time = time.time()

            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                batch = next(dataloader_iterator)

            opt.zero_grad()
            x = batch["net_input"]["src_tokens"].to(self.device)
            y = batch["target"].to(self.device)

            x_mask = (x != padding_idx).unsqueeze(-2)
            y_mask = (y != padding_idx)

            pre_out_x, pre_out_y, mu_theta, sigma_theta = self.model.forward(x, x_mask, y, y_mask)
            loss = self.compute_loss(pre_out_x, pre_out_y, x, y, mu_theta, sigma_theta, vocab_size, step)
            loss.backward()
            opt.step()

            batch_spd = (time.time() - start_time)

            print("Step {:06d}/{:06d} , Loss: {:.2f}, ETA: {:.1f}m".format(
                step + 1,
                self.num_steps,
                loss.item(),
                batch_spd)
            )

            if (step + 1) % self.steps_per_checkpoint == 0:
                if not os.path.exists(self.checkpoints_dir):
                    os.mkdir(self.checkpoints_dir)
                state = {
                    'step': step + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': opt.state_dict(),
                }
                torch.save(state, 'checkpoints/{}-{:06d}'.format(self.model_name, step + 1))

            if (step + 1) % self.steps_per_eval== 0:
                self.eval(self.model, self.dataset_valid, padding_idx, batch_size_eval, step + 1, predictions_dir)

    def eval(self, model, dataset, padding_idx, batch_size_eval, step, predictions_dir):
        print("Evaluating...")
        if not os.path.exists(self.predictions_dir):
            os.mkdir(self.predictions_dir)

        dataloader = DataLoader(dataset, batch_size_eval, collate_fn=dataset.collater)
        file_name = '{}/{}-{:06d}.txt'.format(predictions_dir, self.model_name, step)
        for batch in tqdm(dataloader):
            x = batch["net_input"]["src_tokens"].to(self.device)
            x_mask = (x != padding_idx).unsqueeze(-2)
            pred = self.model.predict(x, x_mask)
            decoded = self.vocab.string(pred).replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
            with open(file_name, 'a') as the_file:
                the_file.write(decoded + '\n')

        # Remove BPE
        output_file_name = '{}/{}-{:06d}-out.txt'.format(predictions_dir, self.model_name, step)
        with open(output_file_name, "w") as file:
            sub = subprocess.run(['sed', '-r', 's/(@@ )|(@@ ?$)//g', file_name], stdout=file)

        with open(output_file_name) as inp, open(output_file_name + ".detok", "w") as out:
            subz = subprocess.run(['perl', 'data/mosesdecoder/scripts/tokenizer/detokenizer.perl', '-q'], stdin=inp, stdout=out)

        # val_path = "data/setimes.tokenized.en-tr/valid.tr"
        scores_file = '{}-scores.txt'.format(self.model_name)
        sacrebleu = subprocess.run(['sacrebleu', '--input', output_file_name+".detok", self.valid_path, '--score-only'], stdout=subprocess.PIPE)
        # print(sacrebleu.stdout.strip())
        bleu_score = sacrebleu.stdout.strip()
        # bleu_score = sacrebleu.stdout.strip()
        with open(scores_file, 'a') as f_score:
            f_score.write("Step {}: {}\n".format(step, bleu_score))


    def compute_loss(self, pre_out_x, pre_out_y, x, y, mu, sigma, vocab_size, step):
        x_stack = torch.stack(pre_out_x, 1).view(-1, vocab_size)
        y_stack = torch.stack(pre_out_y, 1).view(-1, vocab_size)

        x_loss = F.cross_entropy(x_stack, x.long().view(-1))
        y_loss = F.cross_entropy(y_stack, y.long().view(-1))

        KL_loss = self.compute_diagonal_gaussian_kl(mu, sigma)
        if step < self.kl_annealing_steps:
            KL_loss *= step/self.kl_annealing_steps
        return x_loss + y_loss + KL_loss

    def compute_diagonal_gaussian_kl(self, mu, sigma):
        var = sigma ** 2
        loss = torch.mean(- 0.5 * torch.sum(torch.log(var) - mu ** 2 - var, 1))
        return loss
