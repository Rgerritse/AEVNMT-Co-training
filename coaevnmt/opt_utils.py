import torch
import torch.optim as optim
import torch.nn as nn
from itertools import tee


class RequiresGradSwitch:
    """
    Use this to temporarily switch gradient computation on/off for a subset of parameters:
    1. first you requires_grad(value), this will set requires_grad flags to a chosen value (False/True)
        while saving the original value of the flags
    2. then restore(), this will restore requires_grad to its original value
        i.e. whatever requires_grad was before you used requires_grad(value)
    """

    def __init__(self, param_generator):
        self.parameters = param_generator
        self.flags = None

    def requires_grad(self, requires_grad):
        if self.flags is not None:
            raise ValueError("Must restore first")
        self.parameters, parameters = tee(self.parameters, 2)
        flags = []
        for param in parameters:
            flags.append(param.requires_grad)
            param.requires_grad = requires_grad
        self.flags = flags

    def restore(self):
        if self.flags is None:
            raise ValueError("Nothing to restore")
        self.parameters, parameters = tee(self.parameters, 2)
        for param, flag in zip(parameters, self.flags):
            param.requires_grad = flag
        self.flags = None

def create_optimizers(gen_parameters, inf_parameters, config):
    optimizers = {
        "gen": create_optimizer(
            gen_parameters,
            config["opt_type_gen"],
            config["lr_gen"]
        )
    }

    schedulers = {
        "gen": create_scheduler(
            optimizers["gen"],
            config["lr_reduce_factor"],
            config["lr_reduce_patience"],
            config["lr_reduce_cooldown"],
            config["min_lr"]
        )
    }

    if inf_parameters != None:
        optimizers["inf"] = create_optimizer(
            inf_parameters,
            config["opt_type_inf"],
            config["lr_inf"]
        )

        schedulers["inf"] = create_scheduler(
            optimizers["inf"],
            config["lr_reduce_factor"],
            config["lr_reduce_patience"],
            config["lr_reduce_cooldown"],
            config["min_lr"]
        )
    return optimizers, schedulers

def create_optimizer(parameters, type, learning_rate):
    if type is None or type == "adam":
        opt = torch.optim.Adam
    elif type == "adadelta":
        opt = torch.optim.Adadelta
    else:
        raise ValueError("Unknown optimizer: {}".format(type))
    return opt(params=parameters, lr=learning_rate)

def create_scheduler(optimizer, lr_reduce_factor, lr_reduce_patience, lr_reduce_cooldown, min_lr):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        threshold=1e-2,
        threshold_mode="abs",
        cooldown=lr_reduce_cooldown,
        min_lr=min_lr
    )

def optimizer_step(parameters, optimizer, max_gradient_norm):
    if max_gradient_norm > 0:
        nn.utils.clip_grad_norm_(
            parameters,
            max_gradient_norm,
            norm_type=float("inf"))

    optimizer.step()
    optimizer.zero_grad()

def scheduler_step(lr_schedulers, val_bleu):
    for _, lr_scheduler in lr_schedulers.items():
        lr_scheduler.step(val_bleu)
