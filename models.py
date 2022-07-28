"""
Some flax defined machine learning models.
"""


import einops
import flax.linen as nn


class Softmax(nn.Module):
    @nn.compact
    def __call__(self, x, representation=False):
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)


class LeNet_300_100(nn.Module):
    @nn.compact
    def __call__(self, x, representation=False):
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)


class LeNet(nn.Module):
    @nn.compact
    def __call__(self, x, representation=False):
        x = nn.Conv(12, (5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(12, (5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(12, (5, 5), strides=1)(x)
        x = nn.relu(x)
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, representation=False):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if representation:
            return x
        x = nn.Dense(10, name="classifier")(x)
        return nn.softmax(x)
