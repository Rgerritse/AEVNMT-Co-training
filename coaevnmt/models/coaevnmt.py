import torch.nn as nn
import aevnmt_utils as aevnmt_utils
from data_prep import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

class CoAEVNMT(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, config):
        super(CoAEVNMT, self).__init__()
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

        if config["share_embeddings"]:
            self.emb_src = nn.Embedding(vocab_src.size(), config["emb_size"], padding_idx=vocab_src[PAD_TOKEN])
            self.emb_tgt = nn.Embedding(vocab_tgt.size(), config["emb_size"], padding_idx=vocab_tgt[PAD_TOKEN])
        else:
            self.emb_src = None
            self.emb_tgt = None

        self.model_xy = aevnmt_utils.create_model(self.vocab_src, self.vocab_tgt, config, emb_src=self.emb_src, emb_tgt=self.emb_tgt)
        self.model_yx = aevnmt_utils.create_model(self.vocab_tgt, self.vocab_src, config, emb_src=self.emb_tgt, emb_tgt=self.emb_src)

        self.config = config
