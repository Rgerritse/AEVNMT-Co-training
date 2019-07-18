from modules.utils import init_model
from configuration import setup_config

def create_model(vocab_src, vocab_tgt, config):
    if config["model_type"] == "cond_nmt":
        model = cond_nmt_utils.create_model(vocab_src, vocab_tgt, config)
    elif config["model_type"] == "aevnmt":
        model = aevnmt_utils.create_model(vocab_src, vocab_tgt, config)
    else:
        raise ValueError("Unknown model type: {}".format(config["model_type"]))
    print("Model: ", model)
    return model, train_fn, validate_fn

def main():
    config = setup_config()
    train_data, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)

    model = create_model(vocab_src, vocab_tgt, config)
    model.to(torch.device(config["device"]))

    init_model(
        model,
        vocab_src.stoi[config["pad"]],
        vocab_tgt.stoi[config["pad"]],
        config
    )

    if os.path.exists(checkpoints_path):
        checkpoints = [cp for cp in sorted(os.listdir(checkpoints_path)) if '-'.join(cp.split('-')[:-1]) == self.config["session"]]
        if checkpoints:
            state = torch.load('{}/{}'.format(checkpoints_path, checkpoints[-1]))
            self.model.load_state_dict(state['state_dict'])


    

if __name__ == '__main__':
    main()
