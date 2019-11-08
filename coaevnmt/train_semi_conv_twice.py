import os, torch, torchtext

from joeynmt import data
from joeynmt.batch import Batch

from configuration import setup_config
from utils import load_vocabularies, load_data, load_dataset_joey, load_mono_datasets, create_prev, create_optimizer
from torch.nn.utils import clip_grad_norm_
from modules.utils import init_model
import cond_nmt_utils as cond_nmt_utils
import conmt_utils as conmt_utils
import aevnmt_utils as aevnmt_utils
import coaevnmt_utils as coaevnmt_utils
from torch.utils.data import DataLoader
from data_prep import create_batch, BucketingParallelDataLoader, BucketingTextDataLoader
from data_prep.constants import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from itertools import cycle
from data_prep.utils import create_noisy_batch
from opt_utils import RequiresGradSwitch, create_optimizers, optimizer_step, scheduler_step

def create_models(vocab_src, vocab_tgt, config):

    if config["model_type"] == "conmt":
        model_xy = cond_nmt_utils.create_model(vocab_src, vocab_tgt, config)
        model_yx = cond_nmt_utils.create_model(vocab_tgt, vocab_src, config)
        bi_train_fn = conmt_utils.bi_train_fn
        mono_train_fn = conmt_utils.mono_train_fn
        validate_fn = cond_nmt_utils.validate
    elif config["model_type"] == "coaevnmt":
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

def bilingual_step(model_xy, model_yx, sentences_x, sentences_y, bi_train_fn, optimizers_xy, optimizers_yx, vocab_src, vocab_tgt, config, step, device):
    x_in, x_out, x_mask, x_len, x_noisy_in = create_noisy_batch(
        sentences_x, vocab_src, device, word_dropout=config["word_dropout"])
    y_in, y_out, y_mask, y_len, y_noisy_in = create_noisy_batch(
        sentences_y, vocab_tgt, device, word_dropout=config["word_dropout"])

    x_mask = x_mask.unsqueeze(1)
    y_mask = y_mask.unsqueeze(1)

    # Bilingual loss

    bi_loss = bi_train_fn(model_xy, model_yx, x_in, x_noisy_in, x_out, x_len, x_mask, y_in, y_noisy_in, y_out, y_len, y_mask, step)
    bi_loss.backward()

    optimizer_step(model_xy.generative_parameters(), optimizers_xy['gen'], config["max_gradient_norm"])
    optimizer_step(model_yx.generative_parameters(), optimizers_yx['gen'], config["max_gradient_norm"])
    if config["model_type"] == "coaevnmt":
        optimizer_step(model_xy.inference_parameters(), optimizers_xy['inf'], config["max_gradient_norm"])
        optimizer_step(model_yx.inference_parameters(), optimizers_yx['inf'],  config["max_gradient_norm"])

    return bi_loss.item()

def monolingual_step(model_xy, model_yx, cycle_iterate_dl, mono_train_fn, optimizers, vocab_src, vocab_tgt, config, step, device):
    if config["model_type"] == "coaevnmt":
        lm_switch = RequiresGradSwitch(model_xy.lm_parameters())
        if not config["update_lm"]:
            lm_switch.requires_grad(False)

    sentences_y = next(cycle_iterate_dl)

    y_in, y_out, y_mask, y_len, y_noisy_in = create_noisy_batch(
        sentences_y, vocab_tgt, device, word_dropout=config["word_dropout"])

    y_mask = y_mask.unsqueeze(1)

    y_mono_loss = mono_train_fn(model_xy, model_yx, y_in, y_len, y_mask, y_out, vocab_src, config, step)
    y_mono_loss.backward()

    optimizer_step(model_xy.generative_parameters(), optimizers['gen'], config["max_gradient_norm"])
    if config["model_type"] == "coaevnmt":
        optimizer_step(model_xy.inference_parameters(), optimizers['inf'], config["max_gradient_norm"])

        if not config["update_lm"]:  # so we restore switches for source LM
            lm_switch.restore()
    return y_mono_loss.item()

def train(model_xy, model_yx, bi_train_fn, mono_train_fn, validate_fn, bucketing_dl_xy,
    dev_data, cycle_iterate_dl_x, cycle_iterate_dl_y, vocab_src, vocab_tgt, config):

    print("Training...")


    optimizers_xy, schedulers_xy = create_optimizers(
        model_xy.generative_parameters(),
        model_xy.inference_parameters(),
        config
    )

    optimizers_yx, schedulers_yx = create_optimizers(
        model_yx.generative_parameters(),
        model_yx.inference_parameters(),
        config
    )

    saved_epoch = 0
    patience_counter = 0
    max_bleu = 0.0
    converged_counter = 0

    num_batches = sum(1 for _ in iter(bucketing_dl_xy))

    checkpoints_path = "{}/{}/checkpoints".format(config["out_dir"], config["session"])
    if os.path.exists(checkpoints_path):
        checkpoints = [cp for cp in sorted(os.listdir(checkpoints_path)) if cp == config["session"]]
        if checkpoints:
            state = torch.load('{}/{}'.format(checkpoints_path, checkpoints[-1]))
            saved_epoch = state['epoch']
            patience_counter = state['patience_counter']
            max_bleu = state['max_bleu']
            model_xy.load_state_dict(state['state_dict_xy'])
            model_yx.load_state_dict(state['state_dict_yx'])
            optimizers_xy["gen"].load_state_dict(state['optimizer_xy_gen'])
            optimizers_yx["gen"].load_state_dict(state['optimizer_yx_gen'])
            schedulers_xy["gen"].load_state_dict(state['scheduler_xy_gen'])
            schedulers_yx["gen"].load_state_dict(state['scheduler_yx_gen'])
            if config["model_type"] == "coaevnmt":
                optimizers_xy["inf"].load_state_dict(state['optimizer_xy_inf'])
                optimizers_yx["inf"].load_state_dict(state['optimizer_yx_inf'])
                schedulers_xy["inf"].load_state_dict(state['scheduler_xy_inf'])
                schedulers_yx["inf"].load_state_dict(state['scheduler_yx_inf'])
        else:
            init_model(model_xy, vocab_src[PAD_TOKEN], vocab_tgt[PAD_TOKEN], config)
            init_model(model_yx, vocab_tgt[PAD_TOKEN], vocab_src[PAD_TOKEN], config)
    else:
        init_model(model_xy, vocab_src[PAD_TOKEN], vocab_tgt[PAD_TOKEN], config)
        init_model(model_yx, vocab_tgt[PAD_TOKEN], vocab_src[PAD_TOKEN], config)

    curriculum = config["curriculum"].split()
    cycle_iterate_dl_xy = cycle(bucketing_dl_xy)
    cycle_curriculum = cycle(curriculum)
    device = torch.device("cpu") if config["device"] == "cpu" else torch.device("cuda:0")
    for epoch in range(saved_epoch, config["num_epochs"]):
        # Reset optimizers after bilingual warmup
        if epoch == config["bilingual_warmup"] and config["reset_opt"]:
            optimizers_xy, schedulers_xy = create_optimizers(
                model_xy.generative_parameters(),
                model_xy.inference_parameters(),
                config
            )

            optimizers_yx, schedulers_yx = create_optimizers(
                model_yx.generative_parameters(),
                model_yx.inference_parameters(),
                config
            )

        step = 0
        while step < num_batches:
            batch_type = next(cycle_curriculum)
            model_xy.train()
            model_yx.train()
            loss = None
            if batch_type == 'y' and epoch >= config["bilingual_warmup"]:
                loss = monolingual_step(
                        model_xy,
                        model_yx,
                        cycle_iterate_dl_y,
                        mono_train_fn,
                        optimizers_xy,
                        vocab_src,
                        vocab_tgt,
                        config,
                        step,
                        device
                    )
                if not "xy" in curriculum:
                    step += 1
            elif batch_type == 'x' and epoch >= config["bilingual_warmup"]:
                loss = monolingual_step(
                        model_yx,
                        model_xy,
                        cycle_iterate_dl_x,
                        mono_train_fn,
                        optimizers_yx,
                        vocab_tgt,
                        vocab_src,
                        config,
                        step,
                        device
                    )
                if not "xy" in curriculum:
                    step += 1
            elif batch_type == 'xy' or batch_type == 'yx':
                sentences_x, sentences_y = next(cycle_iterate_dl_xy)
                loss = bilingual_step(
                    model_xy,
                    model_yx,
                    sentences_x,
                    sentences_y,
                    bi_train_fn,
                    optimizers_xy,
                    optimizers_yx,
                    vocab_src,
                    vocab_tgt,
                    config,
                    step,
                    device
                )
                step += 1

            # Print progress and loss
            if loss:
                print("Epoch: {:03d}/{:03d}, Batch {:05d}/{:05d}, {}-Loss: {:.2f}".format(
                    epoch + 1,
                    config["num_epochs"],
                    step + 1,
                    num_batches,
                    batch_type,
                    loss
                ))

        val_bleu_xy = evaluate(model_xy, validate_fn, dev_data, vocab_src, vocab_tgt, epoch, config, direction="xy")
        val_bleu_yx = evaluate(model_yx, validate_fn, dev_data, vocab_tgt, vocab_src, epoch, config, direction="yx")

        scheduler_step(schedulers_xy, val_bleu_xy)
        scheduler_step(schedulers_yx, val_bleu_yx)

        print("Blue scores: {}-{}: {}, {}-{}: {}".format(config["src"], config["tgt"], val_bleu_xy, config["tgt"], config["src"], val_bleu_yx))

        if epoch >= config["bilingual_warmup"]:
            if float(val_bleu_xy * val_bleu_yx) > max_bleu:
                max_bleu = float(val_bleu_xy * val_bleu_yx)
                patience_counter = 0

                # Save checkpoint
                if not os.path.exists(checkpoints_path):
                    os.makedirs(checkpoints_path)
                state = {
                    'epoch': epoch + 1,
                    'patience_counter': patience_counter,
                    'max_bleu': max_bleu,
                    'state_dict_xy': model_xy.state_dict(),
                    'state_dict_yx': model_yx.state_dict(),
                    'optimizer_xy_gen': optimizers_xy["gen"].state_dict(),
                    'optimizer_yx_gen': optimizers_yx["gen"].state_dict(),

                    'scheduler_xy_gen': schedulers_xy["gen"].state_dict(),
                    'scheduler_yx_gen': schedulers_yx["gen"].state_dict(),

                }
                if config["model_type"] == "coaevnmt":
                    state['optimizer_xy_inf'] = optimizers_xy["inf"].state_dict()
                    state['optimizer_yx_inf'] = optimizers_yx["inf"].state_dict()
                    state['scheduler_xy_inf'] = schedulers_xy["inf"].state_dict()
                    state['scheduler_yx_inf'] = schedulers_yx["inf"].state_dict()
                torch.save(state, '{}/{}'.format(checkpoints_path, config["session"]))
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    max_bleu = 0
                    patience_counter = 0
                    converged_counter += 1

                    optimizers_xy, schedulers_xy = create_optimizers(
                        model_xy.generative_parameters(),
                        model_xy.inference_parameters(),
                        config
                    )

                    optimizers_yx, schedulers_yx = create_optimizers(
                        model_yx.generative_parameters(),
                        model_yx.inference_parameters(),
                        config
                    )
                    print("Times converged: {}".format(converged_counter))
            if converged_counter >= 2:
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

    vocab_src, vocab_tgt = load_vocabularies(config)
    train_data, dev_data, opt_data = load_data(config, vocab_src=vocab_src, vocab_tgt=vocab_tgt)

    dl_xy = DataLoader(train_data, batch_size=config["batch_size_train"],
                    shuffle=True, num_workers=2)
    bucketing_dl_xy = BucketingParallelDataLoader(dl_xy)

    dl_x = DataLoader(dataset=opt_data['mono_src'], batch_size=config["batch_size_train"], shuffle=True, num_workers=2)
    bucketing_dl_x = BucketingTextDataLoader(dl_x)
    cycle_iterate_dl_x = cycle(bucketing_dl_x)

    dl_y = DataLoader(dataset=opt_data['mono_tgt'], batch_size=config["batch_size_train"], shuffle=True, num_workers=2)
    bucketing_dl_y = BucketingTextDataLoader(dl_y)
    cycle_iterate_dl_y = cycle(bucketing_dl_y)

    model_xy, model_yx, bi_train_fn, mono_train_fn, validate_fn = create_models(vocab_src, vocab_tgt, config)
    model_xy.to(torch.device(config["device"]))
    model_yx.to(torch.device(config["device"]))

    train(
        model_xy,
        model_yx,
        bi_train_fn,
        mono_train_fn,
        validate_fn,
        bucketing_dl_xy,
        dev_data,
        cycle_iterate_dl_x,
        cycle_iterate_dl_y,
        vocab_src,
        vocab_tgt,
        config
    )

if __name__ == '__main__':
    main()
