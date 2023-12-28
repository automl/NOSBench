import torch


def sample_from_prior(
    batch_size=10,
    seq_len=100,
    num_features=1,
):
    ws = torch.distributions.Normal(torch.zeros(num_features + 1), 1.0).sample(
        (batch_size,)
    )

    xs = torch.rand(batch_size, seq_len, num_features)
    ys = torch.distributions.Normal(
        torch.einsum(
            "nmf, nf -> nm", torch.cat([xs, torch.ones(batch_size, seq_len, 1)], 2), ws
        ),
        0.1,
    ).sample()

    return xs, ys


def torch_nanmean(x, axis=0, return_nanshare=False):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    if return_nanshare:
        return value / num, 1.0 - num / x.shape[axis]
    return value / num
