import torch


def build_optimizer(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return optimizer