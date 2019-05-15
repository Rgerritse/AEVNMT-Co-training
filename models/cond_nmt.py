import torch
import torch.nn as nn
import torch.nn.functional as F

class CondNMT(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, encoder, decoder, config):
        super(CondNMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.vocab_tgt = vocab_tgt

        self.emb_src = nn.Embedding(len(vocab_src), config["emb_size"], padding_idx=vocab_src.stoi[config["pad"]])
        self.emb_tgt = nn.Embedding(len(vocab_tgt), config["emb_size"], padding_idx=vocab_tgt.stoi[config["pad"]])

        self.dropout = nn.Dropout(config["dropout"])
        self.logits_layer = nn.Linear(config["hidden_size"], len(vocab_tgt), bias=False)

        self.config = config

    def encode(self, x):
        embed_x = self.dropout(self.emb_src(x))
        enc_output, enc_final = self.encoder(embed_x)
        return enc_output, enc_final

    def forward(self, x, x_mask, y):
        enc_output, enc_final = self.encode(x)
        dec_hidden = self.decoder.initialize(enc_output, enc_final)

        outputs = []
        max_len = y.shape[-1]
        for t in range(max_len):
            prev_y = y[:, t].unsqueeze(1).long()
            embed_y = self.dropout(self.emb_tgt(prev_y))
            pre_output, dec_hidden = self.decoder.forward_step(embed_y, enc_output, x_mask, dec_hidden)
            logits = self.logits_layer(pre_output)
            outputs.append(logits)
        return torch.cat(outputs, dim=1)

    def loss(self, logits, targets, reduction):
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, targets, ignore_index=self.vocab_tgt.stoi[self.config["pad"]], reduction=reduction)
        return loss
