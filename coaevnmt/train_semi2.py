import os, torch, torchtext

from joeynmt import data
from joeynmt.batch import Batch

from configuration import setup_config
from utils import load_dataset_joey, load_mono_datasets
import aevnmt_utils as aevnmt_utils
import coaevnmt_utils as coaevnmt_utils
from modules.utils import init_model
from utils import create_prev
from torch.nn.utils import clip_grad_norm_

def create_models(vocab_src, vocab_tgt, config):
    if config["model_type"] == "coaevnmt":
        model_xy = aevnmt_utils.create_model(vocab_src, vocab_tgt, config)
        model_yx = aevnmt_utils.create_model(vocab_tgt, vocab_src, config)
        bi_train_fn = coaevnmt_utils.bi_train_fn
        mono_train_fn = coaevnmt_utils.mono_train_fn
        validate_fn = aevnmt_utils.validate
    else:
        raise ValueError("Unknown model type: {}".format(config["model_type"]))
    print("Model xy: ", model_xy)
    print("Model yx: ", model_yx)
    return model_xy, model_yx, bi_train_fn, mono_train_fn, validate_fn

def optimizer_step(parameters, optimizer, max_gradient_norm):
    if max_gradient_norm > 0:
        clip_grad_norm_(parameters, max_gradient_norm)

    optimizer.step()
    optimizer.zero_grad()

def create_optimizer(model, config):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(parameters, lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=config["lr_reduce_factor"],
        patience=config["lr_reduce_patience"],
        threshold=1e-2,
        threshold_mode="abs",
        cooldown=config["lr_reduce_cooldown"],
        min_lr=config["min_lr"]
    )

    return opt, scheduler

def train(model_xy, model_yx, bi_train_fn, mono_train_fn, validate_fn, dataloader,
    dev_data, src_mono_buck, tgt_mono_buck, vocab_src, vocab_tgt, config):
    src_sos_idx = vocab_src.stoi[config["sos"]]
    src_eos_idx = vocab_src.stoi[config["eos"]]
    src_pad_idx = vocab_src.stoi[config["pad"]]
    src_unk_idx = vocab_src.stoi[config["unk"]]

    tgt_sos_idx = vocab_tgt.stoi[config["sos"]]
    tgt_eos_idx = vocab_tgt.stoi[config["eos"]]
    tgt_pad_idx = vocab_tgt.stoi[config["pad"]]
    tgt_unk_idx = vocab_tgt.stoi[config["unk"]]

    print("Training...")

    opt_xy, sched_xy = create_optimizer(model_xy, config)
    opt_yx, sched_yx = create_optimizer(model_yx, config)

    saved_epoch = 0
    patience_counter = 0
    max_bleu = 0.0

    checkpoints_path = "{}/{}/checkpoints".format(config["out_dir"], config["session"])
    if os.path.exists(checkpoints_path):
        checkpoints = [cp for cp in sorted(os.listdir(checkpoints_path)) if '-'.join(cp.split('-')[:-1]) == config["session"]]
        if checkpoints:
            state = torch.load('{}/{}'.format(checkpoints_path, checkpoints[-1]))
            saved_epoch = state['epoch']
            patience_counter = state['patience_counter']
            max_bleu = state['max_bleu']
            model_xy.load_state_dict(state['state_dict_xy'])
            model_yx.load_state_dict(state['state_dict_yx'])
            opt_xy.load_state_dict(state['optimizer_xy'])
            opt_yx.load_state_dict(state['optimizer_yx'])
            sched_xy.load_state_dict(state['scheduler_xy'])
            sched_yx.load_state_dict(state['scheduler_yx'])

    tgt_mono_iter = iter(tgt_mono_buck)
    src_mono_iter = iter(src_mono_buck)
    cuda = False if config["device"] == "cpu" else True

    for epoch in range(saved_epoch, config["num_epochs"]):
        for step, batch in enumerate(dataloader):
            model_xy.train()
            model_yx.train()

            opt_xy.zero_grad()
            opt_yx.zero_grad()

            batch = Batch(batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)

            x = batch.src
            prev_x, x_mask = create_prev(x, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])

            y = batch.trg
            prev_y = batch.trg_input
            y_mask = (prev_y != vocab_tgt.stoi[config["pad"]]).unsqueeze(-2)

            if config["word_dropout"] > 0:
                probs_prev_x = torch.zeros(prev_x.shape).uniform_(0, 1).to(prev_x.device)
                prev_x = torch.where(
                    (probs_prev_x > config["word_dropout"]) | (prev_x == src_pad_idx) | (prev_x == src_eos_idx),
                    prev_x,
                    torch.empty(prev_x.shape, dtype=torch.int64).fill_(src_unk_idx).to(prev_x.device)
                )
                probs_prev_y = torch.zeros(prev_y.shape).uniform_(0, 1).to(prev_y.device)
                prev_y = torch.where(
                    (probs_prev_y > config["word_dropout"]) | (prev_y == tgt_pad_idx) | (prev_y == tgt_eos_idx),
                    prev_y,
                    torch.empty(prev_y.shape, dtype=torch.int64).fill_(tgt_unk_idx).to(prev_y.device)
                )

            # Bilingual loss
            bi_loss = bi_train_fn(model_xy, model_yx, prev_x, x, x_mask, prev_y, y, y_mask, step)
            bi_loss.backward()

            optimizer_step(model_xy.parameters(), opt_xy, config["max_gradient_norm"])
            optimizer_step(model_yx.parameters(), opt_yx, config["max_gradient_norm"])

            print_string = "Epoch: {:03d}/{:03d}, Batch {:05d}/{:05d}, Bi-Loss: {:.2f}".format(
                epoch + 1,
                config["num_epochs"],
                step + 1,
                len(dataloader),
                bi_loss.item())

            # Monolingual loss
            if epoch >= config["bilingual_warmup"]:
                # Y data
                try:
                    y_mono_batch = next(tgt_mono_iter)
                except:
                    tgt_mono_iter = iter(tgt_mono_buck)
                    y_mono_batch = next(tgt_mono_iter)
                y_mono_batch = Batch(y_mono_batch, vocab_tgt.stoi[config["pad"]], use_cuda=cuda)
                y_mono = y_mono_batch.src

                prev_y_mono, y_mono_mask = create_prev(y_mono, vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["pad"]])
                mono_y_loss = mono_train_fn(model_xy, model_yx, prev_y_mono, y_mono_mask, y_mono, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]], step)
                mono_y_loss.backward()

                optimizer_step(model_xy.parameters(), opt_xy, config["max_gradient_norm"])

                # X data
                try:
                    x_mono_batch = next(src_mono_iter)
                except:
                    src_mono_iter = iter(src_mono_buck)
                    x_mono_batch = next(src_mono_iter)
                x_mono_batch = Batch(x_mono_batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)
                x_mono = x_mono_batch.src

                prev_x_mono, x_mono_mask = create_prev(x_mono, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])
                mono_x_loss = mono_train_fn(model_yx, model_xy, prev_x_mono, x_mono_mask, x_mono, vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["pad"]], step)
                mono_x_loss.backward()

                optimizer_step(model_yx.parameters(), opt_yx, config["max_gradient_norm"])

                print_string += ", Y-Loss: {:.2f}, X-Loss: {:.2f}".format(mono_y_loss.item(), mono_x_loss.item())
            print(print_string)

        val_bleu_xy = evaluate(model_xy, validate_fn, dataset_dev, vocab_src, vocab_tgt, epoch, config, direction="xy")
        val_bleu_yx = evaluate(model_yx, validate_fn, dataset_dev, vocab_tgt, vocab_src, epoch, config, direction="yx")

        sched_xy.step(float(val_bleu_xy))
        sched_yx.step(float(val_bleu_yx))

        # Save checkpoint
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        state = {
            'epoch': epoch + 1,
            'patience_counter': patience_counter,
            'max_bleu': max_bleu,
            'state_dict_xy': model_xy.state_dict(),
            'state_dict_xy': model_xy.state_dict(),
            'optimizer_xy': opt_xy.state_dict(),
            'optimizer_yx': opt_yx.state_dict(),
            'scheduler_xy': scheduler_xy.state_dict(),
            'scheduler_yx': scheduler_yx.state_dict()
        }
        torch.save(state, '{}/{}-{:03d}'.format(checkpoints_path, config["session"], epoch + 1))

        print("Blue scores: {}, {}".format(val_bleu_xy, val_bleu_yx))
        if float(val_bleu_xy * val_bleu_yx) > max_bleu:
            max_bleu = float(val_bleu)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break

def evaluate(model, validate_fn, dataset_dev, vocab_src, vocab_tgt, epoch, config, direction="xy"):
    checkpoints_path = "{}/{}/checkpoints".format(config["out_dir"], config["session"])
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    model.eval()
    val_bleu = validate_fn(
        model,
        dataset_dev,
        vocab_src,
        vocab_tgt,
        epoch + 1,
        config,
        direction=direction
    )
    return val_bleu

def main():
    config = setup_config()
    train_data, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    train_src_mono, train_tgt_mono = load_mono_datasets(config, vocab_src, vocab_tgt)

    dataloader = data.make_data_iter(train_data, config["batch_size_train"], train=True)
    src_mono_buck = torchtext.data.BucketIterator(train_src_mono, batch_size=config["batch_size_train"], train=True)
    tgt_mono_buck = torchtext.data.BucketIterator(train_tgt_mono, batch_size=config["batch_size_train"], train=True)

    model_xy, model_yx, bi_train_fn, mono_train_fn, validate_fn = create_models(vocab_src, vocab_tgt, config)
    model_xy.to(torch.device(config["device"]))
    model_yx.to(torch.device(config["device"]))

    init_model(model_xy, vocab_src.stoi[config["pad"]], vocab_tgt.stoi[config["pad"]], config)
    init_model(model_yx, vocab_tgt.stoi[config["pad"]], vocab_src.stoi[config["pad"]], config)

    train(model_xy, model_yx, bi_train_fn, mono_train_fn, validate_fn, dataloader,
        dev_data, src_mono_buck, tgt_mono_buck, vocab_src, vocab_tgt, config)

if __name__ == '__main__':
    main()
