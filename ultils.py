import torch
import torch.nn.functional as F
import numpy as np


def nt_xent_loss(out_1, out_2, temperature=1.0):
    batch_size = out_1.shape[0]

    # ROOT FIX: Normalize the raw vectors BEFORE any dot products!
    # This prevents torch.exp() from blowing up to Infinity.
    out_1 = F.normalize(out_1, p=2, dim=-1)
    out_2 = F.normalize(out_2, p=2, dim=-1)

    out = torch.cat([out_1, out_2], dim=0)

    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    mask = ~torch.eye(2 * batch_size, device=out.device).bool()
    negatives = sim.masked_select(mask).view(2 * batch_size, -1)

    positives = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)

    # Add epsilons to prevent divide-by-zero and log(0)
    loss = -torch.log(positives / (negatives.sum(dim=-1) + 1e-8) + 1e-8)
    return loss.mean()


def kl_divergence(mu, logvar):
    # Sum over the latent dimension (-1) rather than the time sequence
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def feature_matching_loss(fake_features, real_features):
    mean_fake = torch.mean(fake_features, dim=0)
    mean_real = torch.mean(real_features, dim=0)
    std_fake = torch.sqrt(torch.var(fake_features, dim=0) + 1e-6)
    std_real = torch.sqrt(torch.var(real_features, dim=0) + 1e-6)

    loss_mean = torch.mean(torch.abs(mean_fake - mean_real))
    loss_std = torch.mean(torch.abs(std_fake - std_real))
    return loss_mean + loss_std


def renormlizer(data, max_val, min_val):
    data = data * max_val
    data = data + min_val
    return data


def np_rounding(prob):
    y = np.round(prob)
    return y