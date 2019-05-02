import torch

def make_init_state(init_state, rnn_type):
    if rnn_type == "lstm":
        h_0 = torch.zeros_like(init_state)
        c_0 = init_state
        return (h_0, c_0)
    elif rnn_type == "gru":
        return init_state
    else:
        raise ValueError("Unknown rnn_type: {} for make_init_state".format(rnn_type))
    return init_state
