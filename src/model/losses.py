import torch
import torch.nn.functional as F


def d_logistic(d_real, d_fake):
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()


def g_logistic(d_fake):
    return F.softplus(-d_fake).mean()


def r1_regularizer(d_real, real_imgs):
    grad_real = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_imgs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_penalty = grad_real.pow(2).view(grad_real.size(0), -1).sum(1).mean()
    return grad_penalty


def path_length_regularizer(G, z, y, mean_path_length, decay=0.01):
    fake = G(z, y)
    noise = torch.randn_like(fake) / (fake.shape[2] * fake.shape[3]) ** 0.5
    grad = torch.autograd.grad(
        outputs=(fake * noise).sum(),
        inputs=z,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    path_lengths = grad.pow(2).sum(1).sqrt()
    mean_path = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    return (path_lengths - mean_path).pow(2).mean(), mean_path.detach()
