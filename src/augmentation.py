import numpy as np


def scaling(x, sigma=1.1):
    # x should be [channel, length]
    if len(x.shape) == 3:  # If batch dimension present
        x = x.squeeze(0)
    factor = np.random.normal(loc=2.0, scale=sigma, size=x.shape[1])
    return x * factor


def permutation(x, max_segments=5):
    # x should be [channel, length]
    if len(x.shape) == 3:  # If batch dimension present
        x = x.squeeze(0)
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments)

    if num_segs > 1:
        split_points = np.random.choice(x.shape[1] - 2, num_segs - 1, replace=False)
        split_points.sort()
        splits = np.split(orig_steps, split_points + 1)
        warp = np.concatenate([splits[p] for p in np.random.permutation(len(splits))])
        return x[:, warp]
    return x


def jitter(x, sigma=0.8):
    # x should be [channel, length]
    if len(x.shape) == 3:  # If batch dimension present
        x = x.squeeze(0)
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)
