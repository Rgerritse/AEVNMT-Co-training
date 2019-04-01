import os
import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, vocab, model, dataset_train, dataset_valid, model_name, num_epochs, device):
        self.vocab = vocab
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.device = device

    def run_epochs(self, padding_idx, vocab_size, batch_size, batch_size_eval, predictions_dir):
        dataloader = DataLoader(self.dataset_train, batch_size, collate_fn=self.dataset_train.collater)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        opt = torch.optim.Adam(parameters)

        saved_epoch = 0
        checkpoints = [cp for cp in sorted(os.listdir('checkpoints')) if '-'.join(cp.split('-')[:-1]) == self.model_name]
        if checkpoints:
            state = torch.load('checkpoints/{}'.format(checkpoints[-1]))
            saved_epoch = state['epoch']
            self.model.load_state_dict(state['state_dict'])
            opt.load_state_dict(state['optimizer'])

        num_batches = len(dataloader)
        for epoch in range(saved_epoch, self.num_epochs):
            start_time = time.time()
            for i, batch in enumerate(dataloader):
                opt.zero_grad()
                x = batch["net_input"]["src_tokens"].to(self.device)
                y = batch["target"].to(self.device)

                x_mask = (x != padding_idx).unsqueeze(-2)
                y_mask = (y != padding_idx)

                pre_out_x, pre_out_y, mu_theta, sigma_theta = self.model.forward(x, x_mask, y, y_mask)
                loss = self.compute_loss(pre_out_x, pre_out_y, x, y, mu_theta, sigma_theta, vocab_size)
                loss.backward()
                opt.step()

                avg_batch_spd = (time.time() - start_time)/(i+1)
                batchs_remaining = num_batches-(i+1)
                ETA = (batchs_remaining * avg_batch_spd)/60

                print("Epoch {:02d}/{:02d}, Batch {:06d}/{:06d} , Loss: {:.2f}, ETA: {:.1f}m".format(
                    epoch +1,
                    self.num_epochs,
                    i + 1,
                    len(dataloader),
                    loss.item(),
                    ETA)
                )

            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(state, 'checkpoints/{}-{:02d}'.format(self.model_name, epoch + 1))
            self.beam_search(self.model, self.dataset_valid, padding_idx, batch_size_eval, epoch + 1, predictions_dir)

    def beam_search(self, model, dataset, padding_idx, batch_size_eval, epoch, predictions_dir):
        dataloader = DataLoader(dataset, batch_size_eval, collate_fn=dataset.collater)
        for i, batch in enumerate(dataloader):
            x = batch["net_input"]["src_tokens"].to(self.device)
            x_mask = (x != padding_idx).unsqueeze(-2)
            pred = self.model.predict(x, x_mask)
            for i in range(pred.shape[0]):
                decoded = self.vocab.string(pred[i])
                with open('{}/{}-{}.txt'.format(predictions_dir, self.model_name, epoch), 'a') as the_file:
                    the_file.write(decoded + '\n')

    def compute_loss(self, pre_out_x, pre_out_y, x, y, mu, sigma, vocab_size):
        x_stack = torch.stack(pre_out_x, 1).view(-1, vocab_size)
        y_stack = torch.stack(pre_out_y, 1).view(-1, vocab_size)

        x_loss = F.cross_entropy(x_stack, x.long().view(-1))
        y_loss = F.cross_entropy(y_stack, y.long().view(-1))

        KL_loss = self.compute_diagonal_gaussian_kl(mu, sigma)

        return x_loss + y_loss + KL_loss

    def compute_diagonal_gaussian_kl(self, mu, sigma):
        var = sigma ** 2
        loss = torch.mean(- 0.5 * torch.sum(torch.log(var) - mu ** 2 - var, 1))
        return loss
