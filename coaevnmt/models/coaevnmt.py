import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from modules.utils import tile_rnn_hidden

class COAEVNMT(nn.Module):
    def __init__(self, vocab_src, vocab_tgt,
        src_inference_model, src_encoder, tgt_decoder, src_language_model,
        tgt_inference_model, tgt_encoder, src_decoder, tgt_language_model,
        config):

        super(COAEVNMT, self).__init__()
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

        # src2tgt modules
        self.src_inference_model = src_inference_model
        self.src_encoder = src_encoder
        self.tgt_decoder = tgt_decoder
        self.src_language_model = src_language_model

        # tgt2src modules
        self.tgt_inference_model = tgt_inference_model
        self.tgt_encoder = tgt_encoder
        self.src_decoder = src_decoder
        self.tgt_language_model = tgt_language_model

        self.src_enc_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])
        self.tgt_dec_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])
        self.src_lm_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])

        self.tgt_enc_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])
        self.src_dec_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])
        self.tgt_lm_init_layer = nn.Linear(config["latent_size"], config["hidden_size"])

        self.emb_src = nn.Embedding(len(vocab_src), config["emb_size"], padding_idx=vocab_src.stoi[config["pad"]])
        self.emb_tgt = nn.Embedding(len(vocab_tgt), config["emb_size"], padding_idx=vocab_tgt.stoi[config["pad"]])

        self.dropout = nn.Dropout(config["dropout"])
        self.config = config

        if not config["tied_embeddings"]:
            self.tm_logits_matrix = nn.Parameter(torch.randn(len(vocab_tgt), config["hidden_size"]))
            self.lm_logits_matrix = nn.Parameter(torch.randn(len(vocab_src), config["hidden_size"]))

        self.register_buffer("prior_loc", torch.zeros([config["latent_size"]]))
        self.register_buffer("prior_scale", torch.ones([config["latent_size"]]))

#===============================================================================
# Source to target functions, model1
#===============================================================================

    def src_inference(self, x, x_mask):
        embed_x = self.emb_src(x).detach()
        loc, scale = self.src_inference_model(embed_x, x_mask)
        return Normal(loc=loc, scale=scale)

    def encode_src(self, x, z):
        embed_x = self.dropout(self.emb_src(x))
        hidden = torch.tanh(self.src_enc_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.src_encoder.rnn)
        enc_output, enc_final = self.src_encoder(embed_x, hidden)
        return enc_output, enc_final

    def init_tgt_decoder(self, encoder_outputs, encoder_final, z):
        self.tgt_decoder.initialize(encoder_outputs, encoder_final)
        hidden = torch.tanh(self.tgt_dec_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.tgt_decoder.rnn)
        return hidden

    def decode_tgt(self, y, enc_output, x_mask, dec_hidden):
        embed_y = self.dropout(self.emb_tgt(y))
        logits = self.tgt_decoder(embed_y, enc_output, x_mask, dec_hidden)
        return logits

    def model_src_language(self, x, z):
        embed_x = self.dropout(self.emb_src(x))
        hidden = torch.tanh(self.src_lm_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.src_language_model.rnn)
        logits = self.src_language_model(embed_x, hidden)
        return logits

    def forward_src2tgt(self, x, x_mask, y, z):
        enc_output, enc_final = self.encode_src(x, z)
        dec_hidden = self.init_tgt_decoder(enc_output, enc_final, z)
        tm_logits = self.decode_tgt(y, enc_output, x_mask, dec_hidden)
        lm_logits = self.model_src_language(x, z)
        return tm_logits, lm_logits

#===============================================================================
# Target 2 source functions, model2
#===============================================================================

    def tgt_inference(self, y, y_mask):
        embed_y = self.emb_tgt(y).detach()
        loc, scale = self.tgt_inference_model(embed_y, y_mask)
        return Normal(loc=loc, scale=scale)

    def encode_tgt(self, y, z):
        embed_y = self.dropout(self.emb_tgt(y))
        hidden = torch.tanh(self.tgt_enc_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.tgt_encoder.rnn)
        enc_output, enc_final = self.tgt_encoder(embed_y, hidden)
        return enc_output, enc_final

    def init_src_decoder(self, encoder_outputs, encoder_final, z):
        self.src_decoder.initialize(encoder_outputs, encoder_final)
        hidden = torch.tanh(self.src_dec_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.src_decoder.rnn)
        return hidden

    def decode_src(self, x, enc_output, y_mask, dec_hidden):
        embed_x = self.dropout(self.emb_src(x))
        logits = self.src_decoder(embed_x, enc_output, y_mask, dec_hidden)
        return logits

    def model_tgt_language(self, y, z):
        embed_y = self.dropout(self.emb_tgt(y))
        hidden = torch.tanh(self.tgt_lm_init_layer(z))
        hidden = tile_rnn_hidden(hidden, self.tgt_language_model.rnn)
        logits = self.tgt_language_model(embed_y, hidden)
        return logits

    def forward_tgt2src(self, y, y_mask, x, z):
        enc_output, enc_final = self.encode_tgt(y, z)
        dec_hidden = self.init_src_decoder(enc_output, enc_final, z)
        tm_logits = self.decode_src(x, enc_output, y_mask, dec_hidden)
        lm_logits = self.model_tgt_language(y, z)
        return tm_logits, lm_logits

#===============================================================================
# Monolingual fuctions
#===============================================================================

    def sample_src(self, y, y_mask):
        with torch.no_grad():
            qz = self.tgt_inference(y, y_mask)
            z = qz.sample()

            enc_output, enc_final = self.encode_tgt(y, z)
            dec_hidden = self.init_src_decoder(enc_output, enc_final, z)
            x = self.src_decoder.sample(self.emb_src, enc_output, y_mask, dec_hidden, self.vocab_src.stoi[self.config["sos"]])
            return x, qz

    def sample_tgt(self, x, x_mask):
        with torch.no_grad():
            qz = self.src_inference(x, x_mask)
            z = qz.sample()

            enc_output, enc_final = self.encode_src(x, z)
            dec_hidden = self.init_tgt_decoder(enc_output, enc_final, z)
            y = self.tgt_decoder.sample(self.emb_tgt, enc_output, x_mask, dec_hidden, self.vocab_tgt.stoi[self.config["sos"]])
            return y, qz


#===============================================================================
# Loss
#===============================================================================
    def loss_fake(self, tm_logits, lm_logits, tm_targets, lm_targets, qz, step):
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

    def loss(self, tm1_logits, lm1_logits, qz1,
        tm2_logits, lm2_logits, qz2, y_targets, x_targets,
        tm3_logits, lm3_logits, tm3_targets, lm3_targets, qz3,
        tm4_logits, lm4_logits, tm4_targets, lm4_targets, qz4, step):

        kl_weight = 1.0
        if (self.config["kl_annealing_steps"] > 0 and step < self.config["kl_annealing_steps"]):
            kl_weight *= 0.001 + (1.0-0.001) / self.config["kl_annealing_steps"] * step

        # Bilingual src2tgt loss
        tm1_logits = tm1_logits.permute(0, 2, 1)
        tm1_loss = F.cross_entropy(tm1_logits, y_targets, ignore_index=self.vocab_tgt.stoi[self.config["pad"]], reduction="none")
        tm1_loss = tm1_loss.sum(dim=1)

        lm1_logits = lm1_logits.permute(0, 2, 1)
        lm1_loss = F.cross_entropy(lm1_logits, x_targets, ignore_index=self.vocab_src.stoi[self.config["pad"]], reduction="none")
        lm1_loss = lm1_loss.sum(dim=1)

        pz1 = torch.distributions.Normal(loc=self.prior_loc, scale=self.prior_scale).expand(qz1.mean.size())
        kl1_loss = torch.distributions.kl.kl_divergence(qz1, pz1)
        kl1_loss = kl1_loss.sum(dim=1)
        kl1_loss *= kl_weight
        if (self.config["kl_free_nats"] > 0 and (self.config["kl_annealing_steps"] == 0 or step >=self.config["kl_annealing_steps"])):
            kl1_loss = torch.clamp(kl1_loss, min=self.config["kl_free_nats"])

        tm1_log_likelihood = -tm1_loss
        lm1_log_likelihood = -lm1_loss
        elbo1 = tm1_log_likelihood + lm1_log_likelihood - kl1_loss
        loss1 = -elbo1

        # Bilingual tgt2src loss
        tm2_logits = tm2_logits.permute(0, 2, 1)
        tm2_loss = F.cross_entropy(tm2_logits, x_targets, ignore_index=self.vocab_src.stoi[self.config["pad"]], reduction="none")
        tm2_loss = tm2_loss.sum(dim=1)

        lm2_logits = lm2_logits.permute(0, 2, 1)
        lm2_loss = F.cross_entropy(lm2_logits, y_targets, ignore_index=self.vocab_tgt.stoi[self.config["pad"]], reduction="none")
        lm2_loss = lm2_loss.sum(dim=1)

        pz2 = torch.distributions.Normal(loc=self.prior_loc, scale=self.prior_scale).expand(qz2.mean.size())
        kl2_loss = torch.distributions.kl.kl_divergence(qz2, pz2)
        kl2_loss = kl2_loss.sum(dim=1)
        kl2_loss *= kl_weight
        if (self.config["kl_free_nats"] > 0 and (self.config["kl_annealing_steps"] == 0 or step >=self.config["kl_annealing_steps"])):
            kl2_loss = torch.clamp(kl2_loss, min=self.config["kl_free_nats"])

        tm2_log_likelihood = -tm2_loss
        lm2_log_likelihood = -lm2_loss
        elbo2 = tm2_log_likelihood + lm2_log_likelihood - kl2_loss
        loss2 = -elbo2

        # Monolingual tgt loss
        tm3_logits = tm3_logits.permute(0, 2, 1)
        tm3_loss = F.cross_entropy(tm3_logits, tm3_targets, ignore_index=self.vocab_tgt.stoi[self.config["pad"]], reduction="none")
        tm3_loss = tm3_loss.sum(dim=1)

        lm3_logits = lm3_logits.permute(0, 2, 1)
        lm3_loss = F.cross_entropy(lm3_logits, lm3_targets, ignore_index=self.vocab_src.stoi[self.config["pad"]], reduction="none")
        lm3_loss = lm3_loss.sum(dim=1)

        pz3 = torch.distributions.Normal(loc=self.prior_loc, scale=self.prior_scale).expand(qz3.mean.size())
        kl3_loss = torch.distributions.kl.kl_divergence(qz3, pz3)
        kl3_loss = kl3_loss.sum(dim=1)
        kl3_loss *= kl_weight
        if (self.config["kl_free_nats"] > 0 and (self.config["kl_annealing_steps"] == 0 or step >=self.config["kl_annealing_steps"])):
            kl3_loss = torch.clamp(kl3_loss, min=self.config["kl_free_nats"])

        tm3_log_likelihood = -tm3_loss
        lm3_log_likelihood = -lm3_loss
        elbo3 = tm3_log_likelihood + lm3_log_likelihood - kl3_loss
        loss3 = -elbo3

        # Monolingual src loss
        tm4_logits = tm4_logits.permute(0, 2, 1)
        tm4_loss = F.cross_entropy(tm4_logits, tm4_targets, ignore_index=self.vocab_src.stoi[self.config["pad"]], reduction="none")
        tm4_loss = tm4_loss.sum(dim=1)

        lm4_logits = lm4_logits.permute(0, 2, 1)
        lm4_loss = F.cross_entropy(lm4_logits, lm4_targets, ignore_index=self.vocab_tgt.stoi[self.config["pad"]], reduction="none")
        lm4_loss = lm4_loss.sum(dim=1)

        pz4 = torch.distributions.Normal(loc=self.prior_loc, scale=self.prior_scale).expand(qz4.mean.size())
        kl4_loss = torch.distributions.kl.kl_divergence(qz4, pz4)
        kl4_loss = kl4_loss.sum(dim=1)
        kl4_loss *= kl_weight
        if (self.config["kl_free_nats"] > 0 and (self.config["kl_annealing_steps"] == 0 or step >=self.config["kl_annealing_steps"])):
            kl4_loss = torch.clamp(kl4_loss, min=self.config["kl_free_nats"])

        tm4_log_likelihood = -tm4_loss
        lm4_log_likelihood = -lm4_loss
        elbo4 = tm4_log_likelihood + lm4_log_likelihood - kl4_loss
        loss4 = -elbo4

        loss = loss1.mean() + loss2.mean() + loss3.mean() + loss4.mean()
        return loss
