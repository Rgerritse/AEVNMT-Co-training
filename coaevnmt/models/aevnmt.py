import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np
from modules.utils import tile_rnn_hidden
from data_prep import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from itertools import chain

class AEVNMT(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, inference_model, encoder, decoder, language_model, config):
        super(AEVNMT, self).__init__()
        self.inference_model = inference_model
        self.encoder = encoder
        self.decoder = decoder
        self.language_model = language_model

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

        self.emb_src = nn.Embedding(vocab_src.size(), config["emb_size"], padding_idx=vocab_src[PAD_TOKEN])
        self.emb_tgt = nn.Embedding(vocab_tgt.size(), config["emb_size"], padding_idx=vocab_tgt[PAD_TOKEN])

        self.enc_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])
        self.dec_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])
        self.lm_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])

        if config["z_vocab_loss"]:
            self.z_loss_src_layer = nn.Linear(config["latent_size"], vocab_src.size())
            self.z_loss_tgt_layer = nn.Linear(config["latent_size"], vocab_tgt.size())
            self.bceloss = nn.BCEWithLogitsLoss(reduction="none")

        if not config["tied_embeddings"]:
            self.tm_logits_matrix = nn.Parameter(torch.randn(vocab_tgt.size(), config["hidden_size"]))
            self.lm_logits_matrix = nn.Parameter(torch.randn(vocab_src.size(), config["hidden_size"]))

        self.dropout = nn.Dropout(config["dropout"])
        self.config = config

        self.register_buffer("prior_loc", torch.zeros([config["latent_size"]]))
        self.register_buffer("prior_scale", torch.ones([config["latent_size"]]))

    def inference_parameters(self):
        return self.inference_model.parameters()

    def generative_parameters(self):
        return chain(self.lm_parameters(), self.tm_parameters())

    def lm_parameters(self):
        return chain(
            self.language_model.parameters(),
            self.lm_init_layer.parameters(),
            self.emb_src.parameters()
        )

    def tm_parameters(self):
        params = chain(self.encoder.parameters(),
                     self.decoder.parameters(),
                     self.emb_tgt.parameters(),
                     self.enc_init_layer.parameters(),
                     self.dec_init_layer.parameters()
                )
        if not self.config["tied_embeddings"]:
            params = chain(params,
                self.tm_logits_matrix.parameters(),
                self.lm_logits_matrix.parameters())
        return params

    def src_embed(self, x):
        x_embed = self.emb_src(x)
        x_embed = self.dropout(x_embed)
        return x_embed

    def tgt_embed(self, y):
        y_embed = self.emb_tgt(y)
        y_embed = self.dropout(y_embed)
        return y_embed

    def inference(self, x, x_mask, x_len):
        embed_x = self.emb_src(x).detach()
        loc, scale = self.inference_model(embed_x, x_mask, x_len)
        return Normal(loc=loc, scale=scale)

    def encode(self, x, x_len, z):
        embed_x = self.dropout(self.emb_src(x))
        hidden = torch.tanh(self.enc_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.encoder.rnn)
        enc_output, enc_final = self.encoder(embed_x, x_len, hidden)
        return enc_output, enc_final

    def decode(self, y, enc_output, x_mask, dec_hidden):
        embed_y = self.dropout(self.emb_tgt(y))
        logits = self.decoder(embed_y, enc_output, x_mask, dec_hidden)
        return logits

    def init_decoder(self, encoder_outputs, encoder_final, z):
        self.decoder.initialize(encoder_outputs, encoder_final)
        hidden = torch.tanh(self.dec_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.decoder.rnn)
        return hidden

    def generate_tm(self, pre_output):
        W = self.emb_tgt.weight if self.config["tied_embeddings"] else self.tm_logits_matrix
        return F.linear(pre_output, W)

    def generate_lm(self, pre_output):
        W = self.emb_src.weight if self.config["tied_embeddings"] else self.lm_logits_matrix
        return F.linear(pre_output, W)

    def model_language(self, x, z):
        embed_x = self.dropout(self.emb_src(x))
        hidden = torch.tanh(self.lm_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.language_model.rnn)

        outputs = []
        max_len = embed_x.shape[1]
        for t in range(max_len):
            prev_x = embed_x[:, t:t+1, :]
            pre_output, hidden = self.language_model.forward_step(prev_x, hidden, z)
            logits = self.generate_lm(pre_output)
            outputs.append(logits)
        return torch.cat(outputs, dim=1)

    def forward(self, x, x_len, x_mask, y, z):
        enc_output, enc_final = self.encode(x, x_len, z)
        dec_hidden = self.init_decoder(enc_output, enc_final, z)

        tm_outputs = []
        max_len = y.shape[-1]
        for t in range(max_len):
            prev_y = y[:, t]
            embed_y = self.tgt_embed(prev_y)
            pre_output, dec_hidden = self.decoder.forward_step(embed_y, enc_output, x_mask, dec_hidden, z)
            logits = self.generate_tm(pre_output)
            tm_outputs.append(logits)
        tm_logits = torch.cat(tm_outputs, dim=1)
        lm_logits = self.model_language(x, z)

        z_src_logits = None
        z_tgt_logits = None
        if self.config["z_vocab_loss"]:
            z_src_logits = self.z_loss_src_layer(z)
            z_tgt_logits = self.z_loss_tgt_layer(z)

        return tm_logits, lm_logits, z_src_logits, z_tgt_logits

    # Change this with ancestral_sample
    def sample(self, enc_output, y_mask, dec_hidden):
        batch_size = y_mask.size(0)
        prev = y_mask.new_full(size=[batch_size], fill_value=self.vocab_tgt[SOS_TOKEN],
            dtype=torch.long)

        output = []
        for t in range(self.config["max_len"]):
            embed = self.emb_tgt(prev)
            pre_output, dec_hidden = self.decoder.forward_step(embed, enc_output, y_mask, dec_hidden)
            logits = self.generate_tm(pre_output)
            categorical = Categorical(logits=logits)
            next_word = categorical.sample()
            output.append(next_word.squeeze(1).cpu().numpy())
            prev = next_word.squeeze(1)
            stacked_output = np.stack(output, axis=1)  # batch, time
        return stacked_output

    def loss(self, tm_logits, lm_logits, z_src_logits, z_tgt_logits, tm_targets, lm_targets, qz, step):
        kl_weight = 1.0
        if (self.config["kl_annealing_steps"] > 0 and step < self.config["kl_annealing_steps"]):
            kl_weight *= 0.001 + (1.0-0.001) / self.config["kl_annealing_steps"] * step

        tm_logits = tm_logits.permute(0, 2, 1)
        tm_loss = F.cross_entropy(tm_logits, tm_targets, ignore_index=self.vocab_tgt[PAD_TOKEN], reduction="none")
        tm_loss = tm_loss.sum(dim=1)

        lm_logits = lm_logits.permute(0, 2, 1)
        lm_loss = F.cross_entropy(lm_logits, lm_targets, ignore_index=self.vocab_src[PAD_TOKEN], reduction="none")
        lm_loss = lm_loss.sum(dim=1)

        pz = torch.distributions.Normal(loc=self.prior_loc, scale=self.prior_scale).expand(qz.mean.size())
        kl_loss = torch.distributions.kl.kl_divergence(qz, pz)
        if (self.config["kl_free_nats_style"] == "indv"):
            if (self.config["kl_free_nats"] > 0 and (self.config["kl_annealing_steps"] == 0 or step >= self.config["kl_annealing_steps"])):
                kl_loss = torch.clamp(kl_loss, min=self.config["kl_free_nats"])
        kl_loss = kl_loss.sum(dim=1)

        if (self.config["kl_free_nats_style"] == "sum"):
            if (self.config["kl_free_nats"] > 0 and (self.config["kl_annealing_steps"] == 0 or step >= self.config["kl_annealing_steps"])):
                kl_loss = torch.clamp(kl_loss, min=self.config["kl_free_nats"])

        kl_loss *= kl_weight

        tm_log_likelihood = -tm_loss
        lm_log_likelihood = -lm_loss
        elbo = tm_log_likelihood + lm_log_likelihood - kl_loss
        loss = -elbo

        if self.config["z_vocab_loss"]:
            z_src_target = torch.zeros_like(z_src_logits)
            z_tgt_target = torch.zeros_like(z_tgt_logits)

            index_src = torch.arange(lm_targets.shape[0], dtype=torch.long)[:, None].expand(-1, lm_targets.shape[1])
            z_src_target[index_src.reshape(-1), lm_targets.view(-1)] = 1
            z_src_loss = self.bceloss(z_src_logits, z_src_target).sum(dim=1)
            loss += z_src_loss

            index_tgt = torch.arange(tm_targets.shape[0], dtype=torch.long)[:, None].expand(-1, tm_targets.shape[1])
            z_tgt_target[index_tgt.reshape(-1), tm_targets.view(-1)] = 1
            z_tgt_loss = self.bceloss(z_tgt_logits, z_tgt_target).sum(dim=1)
            loss += z_tgt_loss
        # loss = tm_loss + lm_loss + kl_loss
        return loss.mean()
