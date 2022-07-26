"""
Some flax defined machine learning models.
"""


import einops
import flax.linen as nn


class Softmax(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(10), nn.softmax
            ]
        )(x)


class LeNet_300_100(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(300), nn.relu,
                nn.Dense(100), nn.relu,
                nn.Dense(10), nn.softmax
            ]
        )(x)


class LeNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                nn.Conv(12, (5, 5), strides=2), nn.sigmoid,
                nn.Conv(12, (5, 5), strides=2), nn.sigmoid,
                nn.Conv(12, (5, 5), strides=1), nn.sigmoid,
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(10), nn.softmax,
            ]
        )(x)


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                nn.Conv(32, (3, 3)), nn.relu,
                lambda x: nn.max_pool(x, (2, 2)),
                nn.Conv(64, (3, 3)), nn.relu,
                lambda x: nn.max_pool(x, (2, 2)),
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(100), nn.relu,
                nn.Dense(10), nn.softmax,
            ]
        )(x)
