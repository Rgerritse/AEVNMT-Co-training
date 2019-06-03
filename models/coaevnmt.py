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
