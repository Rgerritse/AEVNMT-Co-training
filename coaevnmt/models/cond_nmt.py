import torch
import torch.nn as nn
import torch.nn.functional as F
from data_prep import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

class CondNMT(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, encoder, decoder, config):
        super(CondNMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.vocab_tgt = vocab_tgt

        self.emb_src = nn.Embedding(vocab_src.size(), config["emb_size"], padding_idx=vocab_src[PAD_TOKEN])
        self.emb_tgt = nn.Embedding(vocab_tgt.size(), config["emb_size"], padding_idx=vocab_tgt[PAD_TOKEN])

        if not config["tied_embeddings"]:
            self.output_matrix = nn.Parameter(torch.randn(len(vocab_tgt), config["hidden_size"]))

        self.dropout_layer = nn.Dropout(config["dropout"])

        self.config = config

    def inference_parameters(self):
        return None
        
    def generative_parameters(self):
        return self.parameters()

    def src_embed(self, x):
        x_embed = self.emb_src(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed

    def tgt_embed(self, y):
        y_embed = self.emb_tgt(y)
        y_embed = self.dropout_layer(y_embed)
        return y_embed

    def generate_tm(self, pre_output):
        W = self.emb_tgt.weight if self.config["tied_embeddings"] else self.output_matrix
        return F.linear(pre_output, W)

    def encode(self, x, x_len):
        x_embed =  self.src_embed(x)
        enc_output, enc_final = self.encoder(x_embed, x_len)
        return enc_output, enc_final

    def init_decoder(self, encoder_outputs, encoder_final):
        return self.decoder.initialize(encoder_outputs, encoder_final)

    def forward(self, x, x_mask, x_len, y):
        enc_output, enc_final = self.encode(x, x_len)
        dec_hidden = self.init_decoder(enc_output, enc_final)

        # Decode function
        outputs = []
        max_len = y.shape[-1]
        for t in range(max_len):
            prev_y = y[:, t]
            embed_y = self.tgt_embed(prev_y)
            pre_output, dec_hidden = self.decoder.forward_step(embed_y, enc_output, x_mask, dec_hidden)
            logits = self.generate_tm(pre_output)
            outputs.append(logits)
        return torch.cat(outputs, dim=1)

    def loss(self, logits, targets):
        logits = logits.permute(0, 2, 1)
        # loss = F.cross_entropy(logits, targets, ignore_index=self.vocab_tgt.stoi[self.config["pad"]], reduction="none")
        loss = F.cross_entropy(logits, targets, ignore_index=self.vocab_tgt[PAD_TOKEN], reduction="none")
        loss = loss.sum(dim=1)
        return loss.mean()
