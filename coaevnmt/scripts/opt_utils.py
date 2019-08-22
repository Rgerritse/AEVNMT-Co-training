import torch

def create_optimizers(gen_parameters, inf_parameters, config):
    optimizers = {
        "gen": create_optimizer(
            gen_parameters,
            config["opt_type_gen"],
            config["lr_gen"]
        ),
        "inf": get_optimizer(
            inf_parameters,
            config["opt_type_inf"],
            config["lr_inf"]
        )
    }

    schedulers = {
        "gen": create_scheduler(
            optimizers["gen"],
            config["lr_reduce_factor"],
            config["lr_reduce_patience"],
            config["lr_reduce_cooldown"],
            config["min_lr"]
        ),
        "inf": create_scheduler(
            optimizers["inf"],
            config["lr_reduce_factor"],
            config["lr_reduce_patience"],
            config["lr_reduce_cooldown"],
            config["min_lr"]
        )

    }
    return optimizers, lr_schedulers

def create_optimizer(parameters, type, learning_rate):
    if name is None or name == "adam":
        opt = torch.optim.Adam
    elif name == "adadelta":
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

def scheduler_step(lr_schedulers, val_bleu, hparams):
    for _, lr_scheduler in lr_schedulers.items():
        lr_scheduler.step(val_bleu)


#
# def take_optimizer_step(optimizer, parameters, clip_grad_norm=0., zero_grad=True):
#     if clip_grad_norm > 0:
#         nn.utils.clip_grad_norm_(parameters=parameters,
#                                  max_norm=clip_grad_norm,
#                                  norm_type=float("inf"))
#     optimizer.step()
#     if zero_grad:
#         optimizer.zero_grad()
# def create_optimizer(parameter)
# def create_optimizer(parameters, config):
#     optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="max",
#         factor=config["lr_reduce_factor"],
#         patience=config["lr_reduce_patience"],
#         threshold=1e-2,
#         threshold_mode="abs",
#         cooldown=config["lr_reduce_cooldown"],
#         min_lr=config["min_lr"]
#     )
#     return optimizer, scheduler
