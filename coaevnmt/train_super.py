import os, torch, time

from joeynmt import data
from joeynmt.batch import Batch

from configuration import setup_config
from utils import load_dataset_joey, create_prev, create_optimizer
from torch.nn.utils import clip_grad_norm_
from modules.utils import init_model
import cond_nmt_utils as cond_nmt_utils
import aevnmt_utils as aevnmt_utils

def create_model(vocab_src, vocab_tgt, config):
    if config["model_type"] == "cond_nmt":
        model = cond_nmt_utils.create_model(vocab_src, vocab_tgt, config)
        train_fn = cond_nmt_utils.train_step
        validate_fn = cond_nmt_utils.validate
    elif config["model_type"] == "aevnmt":
        model = aevnmt_utils.create_model(vocab_src, vocab_tgt, config)
        train_fn = aevnmt_utils.train_step
        validate_fn = aevnmt_utils.validate
    else:
        raise ValueError("Unknown model type: {}".format(config["model_type"]))
    print("Model: ", model)
    return model, train_fn, validate_fn

def train(model, train_fn, validate_fn, dataloader, dev_data, vocab_src, vocab_tgt, config):
    src_sos_idx = vocab_src.stoi[config["sos"]]
    src_eos_idx = vocab_src.stoi[config["eos"]]
    src_pad_idx = vocab_src.stoi[config["pad"]]
    src_unk_idx = vocab_src.stoi[config["unk"]]

    tgt_sos_idx = vocab_tgt.stoi[config["sos"]]
    tgt_eos_idx = vocab_tgt.stoi[config["eos"]]
    tgt_pad_idx = vocab_tgt.stoi[config["pad"]]
    tgt_unk_idx = vocab_tgt.stoi[config["unk"]]

    print("Training...")
    optimizer, scheduler = create_optimizer(model.parameters(), config)

    saved_epoch = 0
    patience_counter = 0
    max_bleu = 0.0

    checkpoints_path = "{}/{}/checkpoints".format(config["out_dir"], config["session"])
    if os.path.exists(checkpoints_path):
        checkpoints = [cp for cp in sorted(os.listdir(checkpoints_path)) if cp == config["session"]]
        if checkpoints:
            state = torch.load('{}/{}'.format(checkpoints_path, checkpoints[-1]))
            saved_epoch = state['epoch']
            patience_counter = state['patience_counter']
            max_bleu = state['max_bleu']
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        else:
            init_model(model, vocab_src.stoi[config["pad"]], vocab_tgt.stoi[config["pad"]], config)
    else:
        init_model(model, vocab_src.stoi[config["pad"]], vocab_tgt.stoi[config["pad"]], config)

    cuda = False if config["device"] == "cpu" else True
    for epoch in range(saved_epoch, config["num_epochs"]):
        for step, batch in enumerate(dataloader):
            model.train()

            batch = Batch(batch, src_pad_idx, use_cuda=cuda)

            optimizer.zero_grad()

            x_out = batch.src
            x_len = batch.src_lengths
            x_in, x_mask = create_prev(x_out, src_sos_idx, src_pad_idx)

            y_out = batch.trg
            y_in = batch.trg_input

            # Word_dropout
            probs_x_in = torch.zeros(x_in.shape).uniform_(0, 1).to(x_in.device)
            x_noisy_in = torch.where(
                (probs_x_in > config["word_dropout"]) | (x_in == src_pad_idx) | (x_in == src_eos_idx),
                x_in,
                torch.empty(x_in.shape, dtype=torch.int64).fill_(src_unk_idx).to(x_in.device)
            )
            probs_y_in = torch.zeros(y_in.shape).uniform_(0, 1).to(y_in.device)
            y_noisy_in = torch.where(
                (probs_y_in > config["word_dropout"]) | (y_in == tgt_pad_idx) | (y_in == tgt_eos_idx),
                y_in,
                torch.empty(y_in.shape, dtype=torch.int64).fill_(tgt_unk_idx).to(y_in.device)
            )

            loss = train_fn(model, x_in, x_noisy_in, x_out, x_len, x_mask, y_in, y_noisy_in, y_out, step)
            loss.backward()

            if config["max_gradient_norm"] > 0:
                clip_grad_norm_(model.parameters(), config["max_gradient_norm"])

            optimizer.step()

            print("Epoch: {:03d}/{:03d}, Batch {:05d}/{:05d}, Loss: {:.2f}".format(
                epoch + 1,
                config["num_epochs"],
                step + 1,
                len(dataloader),
                loss.item())
            )


        val_bleu = evaluate(model, validate_fn, dev_data, vocab_src, vocab_tgt, epoch, config)
        scheduler.step(float(val_bleu))

        print("Blue score: {}".format(val_bleu))
        if float(val_bleu) > max_bleu:
            max_bleu = float(val_bleu)
            patience_counter = 0

            # Save checkpoint
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            state = {
                'epoch': epoch + 1,
                'patience_counter': patience_counter,
                'max_bleu': max_bleu,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, '{}/{}'.format(checkpoints_path, config["session"]))
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break

def evaluate(model, validate_fn, dev_data, vocab_src, vocab_tgt, epoch, config):
    checkpoints_path = "{}/{}/checkpoints".format(config["out_dir"], config["session"])
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    model.eval()
    val_bleu = validate_fn(
        model,
        dev_data,
        vocab_src,
        vocab_tgt,
        epoch + 1,
        config
    )
    return val_bleu



def main():
    config = setup_config()

    train_data, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    dataloader = data.make_data_iter(train_data, config["batch_size_train"], train=True)

    model, train_fn, validate_fn = create_model(vocab_src, vocab_tgt, config)
    model.to(torch.device(config["device"]))

    train(model, train_fn, validate_fn, dataloader, dev_data, vocab_src, vocab_tgt, config)

if __name__ == '__main__':
    main()
