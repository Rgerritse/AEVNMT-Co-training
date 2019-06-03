import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from modules.utils import tile_rnn_hidden

class AEVNMT(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, inference_model, encoder, decoder, language_model, config):
        super(AEVNMT, self).__init__()
        self.inference_model = inference_model
        self.encoder = encoder
        self.decoder = decoder
        self.language_model = language_model

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

        self.emb_src = nn.Embedding(len(vocab_src), config["emb_size"], padding_idx=vocab_src.stoi[config["pad"]])
        self.emb_tgt = nn.Embedding(len(vocab_tgt), config["emb_size"], padding_idx=vocab_tgt.stoi[config["pad"]])

        self.enc_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])
        self.dec_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])
        self.lm_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])

        self.dropout = nn.Dropout(config["dropout"])
        self.config = config

        self.register_buffer("prior_loc", torch.zeros([config["latent_size"]]))
        self.register_buffer("prior_scale", torch.ones([config["latent_size"]]))

    def inference(self, x, x_mask):
        embed_x = self.emb_src(x).detach()
        loc, scale = self.inference_model(embed_x, x_mask)
        return Normal(loc=loc, scale=scale)

    def encode(self, x, z):
        embed_x = self.dropout(self.emb_src(x))
        hidden = torch.tanh(self.enc_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.encoder.rnn)
        enc_output, enc_final = self.encoder(embed_x, hidden)
        return enc_output, enc_final

    def decode(self, y, enc_output, x_mask, dec_hidden):
        embed_y = self.dropout(self.emb_tgt(y))
        logits = self.decoder(embed_y, enc_output, x_mask, dec_hidden)
        return logits

    def model_language(self, x, z):
        embed_x = self.dropout(self.emb_src(x))
        hidden = torch.tanh(self.lm_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.language_model.rnn)
        logits = self.language_model(embed_x, hidden)
        return logits

    def init_decoder(self, encoder_outputs, encoder_final, z):
        self.decoder.initialize(encoder_outputs, encoder_final)
        hidden = torch.tanh(self.dec_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.decoder.rnn)
        return hidden

    def forward(self, x, x_mask, y, z):
        enc_output, enc_final = self.encode(x, z)
        dec_hidden = self.init_decoder(enc_output, enc_final, z)

        outputs = []
        max_len = y.shape[-1]
        for t in range(max_len):
            prev_y = y[:, t].unsqueeze(1).long()
            embed_y = self.dropout(self.emb_tgt(prev_y))
            pre_output, dec_hidden = self.decoder.forward_step(embed_y, enc_output, x_mask, dec_hidden)
            logits = self.decoder.logits_layer(pre_output)
            outputs.append(logits)
        tm_logits = torch.cat(outputs, dim=1)

        # tm_logits = self.decode(y, enc_output, x_mask, dec_hidden)
        lm_logits = self.model_language(x, z)

        return tm_logits, lm_logits

    def loss(self, tm_logits, lm_logits, tm_targets, lm_targets, qz, step):
        tm_logits = tm_logits.permute(0, 2, 1)
        tm_loss = F.cross_entropy(tm_logits, tm_targets, ignore_index=self.vocab_tgt.stoi[self.config["pad"]], reduction="none")
        tm_loss = tm_loss.sum(dim=1)

        lm_logits = lm_logits.permute(0, 2, 1)
        lm_loss = F.cross_entropy(lm_logits, lm_targets, ignore_index=self.vocab_src.stoi[self.config["pad"]], reduction="none")
        lm_loss = lm_loss.sum(dim=1)

        pz = torch.distributions.Normal(loc=self.prior_loc, scale=self.prior_scale).expand(qz.mean.size())
        kl_loss = torch.distributions.kl.kl_divergence(qz, pz)
        kl_loss = kl_loss.sum(dim=1)
        if (self.config["kl_free_nats"] > 0):
            kl_loss = torch.clamp(kl_loss, min=self.config["kl_free_nats"])

        tm_log_likelihood = -tm_loss
        lm_log_likelihood = -lm_loss
        elbo = tm_log_likelihood + lm_log_likelihood - kl_loss
        loss = -elbo
        # loss = tm_loss + lm_loss + kl_loss
        return loss.mean()
