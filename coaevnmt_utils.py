from modules.inference import InferenceModel
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.language import LanguageModel
from models.coaevnmt import COAEVNMT
from modules.search import beam_search
from utils import create_attention, clean_sentences, compute_bleu, save_hypotheses, create_prev_x
from joeynmt import data
from joeynmt.batch import Batch

def create_model(vocab_src, vocab_tgt, config):
    inference_model1 = InferenceModel(config)
    encoder1 = Encoder(config)
    attention1 = create_attention(config)
    decoder1 = Decoder(attention1, len(vocab_tgt), config)
    language_model1 = LanguageModel(len(vocab_src), config)

    inference_model2 = InferenceModel(config)
    encoder2 = Encoder(config)
    attention2 = create_attention(config)
    decoder2 = Decoder(attention2, len(vocab_tgt), config)
    language_model2 = LanguageModel(len(vocab_src), config)

    model = COAEVNMT(vocab_src, vocab_tgt, inference_model1, encoder1, decoder1,
        language_model1, inference_model2, encoder2, decoder2, language_model2,
        config)
    return model

def train_step(model, prev_x, x, x_mask, prev_y, y, y_mask, step):
    # Bilingual src2tgt model1
    qz1 = model.src_inference(prev_x, x_mask)
    z1 = qz1.rsample()
    tm1_logits, lm1_logits = model.forward_src2tgt(prev_x, x_mask, prev_y, z1)
    print("tm1_logits: ", tm1_logits.shape)
    print("lm1_logits: ", lm1_logits.shape)

    # Bilingual tgt2src model2
    qz2 = model.tgt_inference(prev_y, y_mask)
    z2 = qz2.rsample()
    tm2_logits, lm2_logits = model.forward_tgt2src(prev_y, y_mask, prev_x, z2)
    print("tm2_logits: ", tm2_logits.shape)
    print("lm2_logits: ", lm2_logits.shape)

    asd



def validate(model, dataset_dev, vocab_src, vocab_tgt, epoch, config):
    raise NotImplementedError
