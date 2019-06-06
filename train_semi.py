import os, torch, torchtext

from joeynmt import data
from joeynmt.batch import Batch

from configuration import setup_config
from utils import load_dataset_joey, load_mono_datasets
import coaevnmt_utils as coaevnmt_utils
from modules.utils import init_model
from utils import create_prev
from torch.nn.utils import clip_grad_norm_

def create_model(vocab_src, vocab_tgt, config):
    if config["model_type"] == "coaevnmt":
        model = coaevnmt_utils.create_model(vocab_src, vocab_tgt, config)
        train_fn = coaevnmt_utils.train_step
        validate_fn = coaevnmt_utils.validate
    else:
        raise ValueError("Unknown model type: {}".format(config["model_type"]))
    print("Model: ", model)
    return model, train_fn, validate_fn

def train(model, train_fn, validate_fn, dataloader, dataset_dev, src_mono_buck, tgt_mono_buck, vocab_src, vocab_tgt, config):
    src_sos_idx = vocab_src.stoi[config["sos"]]
    src_eos_idx = vocab_src.stoi[config["eos"]]
    src_pad_idx = vocab_src.stoi[config["pad"]]
    src_unk_idx = vocab_src.stoi[config["unk"]]

    tgt_sos_idx = vocab_tgt.stoi[config["sos"]]
    tgt_eos_idx = vocab_tgt.stoi[config["eos"]]
    tgt_pad_idx = vocab_tgt.stoi[config["pad"]]
    tgt_unk_idx = vocab_tgt.stoi[config["unk"]]

    print("Training...")
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

    saved_epoch = 0
    patience_counter = 0
    max_bleu = 0.0
    checkpoints_path = "{}/{}".format(config["out_dir"], config["checkpoints_dir"])
    if os.path.exists(checkpoints_path):
        checkpoints = [cp for cp in sorted(os.listdir(checkpoints_path)) if '-'.join(cp.split('-')[:-1]) == config["session"]]
        if checkpoints:
            state = torch.load('{}/{}'.format(checkpoints_path, checkpoints[-1]))
            saved_epoch = state['epoch']
            patience_counter = state['patience_counter']
            max_bleu = state['max_bleu']
            self.model.load_state_dict(state['state_dict'])
            opt.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])

    tgt_mono_iter = iter(tgt_mono_buck)
    src_mono_iter = iter(src_mono_buck)
    cuda = False if config["device"] == "cpu" else True
    for epoch in range(saved_epoch, config["num_epochs"]):
        for step, batch in enumerate(dataloader):
            model.train()

            batch = Batch(batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)

            x = batch.src
            prev_x, x_mask = create_prev(x, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])

            y = batch.trg
            prev_y = batch.trg_input
            y_mask = (prev_y != vocab_tgt.stoi[config["pad"]]).unsqueeze(-2)

            try:
                y_mono_batch = next(tgt_mono_iter)
            except:
                tgt_mono_iter = iter(tgt_mono_buck)
                y_mono_batch = next(tgt_mono_iter).src[0]
            y_mono_batch = Batch(y_mono_batch, vocab_tgt.stoi[config["pad"]], use_cuda=cuda)
            y_mono = y_mono_batch.src

            prev_y_mono, y_mono_mask = create_prev(y_mono, vocab_tgt.stoi[config["sos"]], vocab_tgt.stoi[config["pad"]])

            try:
                x_mono_batch = next(src_mono_iter)
            except:
                src_mono_iter = iter(src_mono_buck)
                x_mono_batch = next(src_mono_iter).src[0]
            x_mono_batch = Batch(x_mono_batch, vocab_src.stoi[config["pad"]], use_cuda=cuda)
            x_mono = x_mono_batch.src

            prev_x_mono, x_mono_mask = create_prev(x_mono, vocab_src.stoi[config["sos"]], vocab_src.stoi[config["pad"]])

            # Print statements to check sentences
            # print("x: ", vocab_src.array_to_sentence(x[-1].cpu().numpy(), cut_at_eos=False))
            # print("prev_x: ", vocab_src.array_to_sentence(prev_x[-1].cpu().numpy(), cut_at_eos=False))
            # print("y: ", vocab_tgt.array_to_sentence(y[-1].cpu().numpy(), cut_at_eos=False))
            # print("prev_y: ", vocab_tgt.array_to_sentence(prev_y[-1].cpu().numpy(), cut_at_eos=False))

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
                # Should probably do word_dropout on monolingual sentences???



            loss = train_fn(model, prev_x, x, x_mask, prev_y, y, y_mask, prev_y_mono, y_mono, y_mono_mask, prev_x_mono, x_mono, x_mono_mask, src_pad_idx, tgt_pad_idx, step)
            loss.backward()

            if config["max_gradient_norm"] > 0:
                clip_grad_norm_(model.parameters(), config["max_gradient_norm"])

            opt.step()

            print("Epoch: {:03d}/{:03d}, Batch {:05d}/{:05d}, Loss: {:.2f}".format(
                epoch + 1,
                config["num_epochs"],
                step + 1,
                len(dataloader),
                loss.item())
            )

        val_bleu = evaluate(model, validate_fn, dataset_dev, vocab_src, vocab_tgt, epoch, config)
        scheduler.step(float(val_bleu))

        # Save checkpoint
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        state = {
            'epoch': epoch + 1,
            'patience_counter': patience_counter,
            'max_bleu': max_bleu,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, '{}/{}-{:03d}'.format(checkpoints_path, config["session"], epoch + 1))

        print("Blue score: {}".format(val_bleu))
        if float(val_bleu) > max_bleu:
            max_bleu = float(val_bleu)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= self.config["patience"]:
                break


def evaluate(model, validate_fn, dataset_dev, vocab_src, vocab_tgt, epoch, config):
    checkpoints_path = "{}/{}".format(config["out_dir"], config["predictions_dir"])
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    model.eval()
    val_bleu = validate_fn(
        model,
        dataset_dev,
        vocab_src,
        vocab_tgt,
        epoch + 1,
        config
    )
    return val_bleu

def main():
    config = setup_config()
    train_data, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    train_src_mono, train_tgt_mono = load_mono_datasets(config, vocab_src, vocab_tgt)

    dataloader = data.make_data_iter(train_data, config["batch_size_train"], train=True)
    src_mono_buck = torchtext.data.BucketIterator(train_src_mono, batch_size=config["batch_size_train"], train=True)
    tgt_mono_buck = torchtext.data.BucketIterator(train_tgt_mono, batch_size=config["batch_size_train"], train=True)

    # dataloader = data.make_data_iter(train_data, 1, train=True)
    # src_mono_iter = iter(torchtext.data.BucketIterator(train_src_mono, batch_size=1, train=True))
    # tgt_mono_iter = iter(torchtext.data.BucketIterator(train_tgt_mono, batch_size=1, train=True))

    model, train_fn, validate_fn = create_model(vocab_src, vocab_tgt, config)
    model.to(torch.device(config["device"]))

    init_model(model, vocab_src.stoi[config["pad"]], vocab_tgt.stoi[config["pad"]], config)
    train(model, train_fn, validate_fn, dataloader, dev_data, src_mono_buck, tgt_mono_buck, vocab_src, vocab_tgt, config)

if __name__ == '__main__':
    main()
