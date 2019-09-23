from train_super create_model

def main():
    config = setup_config()
    _, dev_data, vocab_src, vocab_tgt = load_dataset_joey(config)
    model, _, validate_fn = create_model(vocab_src, vocab_tgt, config)

    checkpoint_path = "output/aevnmt_word_dropout_0.1/checkpoints/aevnmt_word_dropout_0.1"
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])

if __name__ == '__main__':
    main()
