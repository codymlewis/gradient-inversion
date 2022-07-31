"""
Some flax defined machine learning models.
"""

from typing import Any

import jax.numpy as jnp
import einops
import flax.linen as nn

ModuleDef = Any

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


# ResNetRS50 implementation based on the tensorflow applications version
# Note: The batch norms require specialized training functions, checkout the resnetrs branch

def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return jnp.pad(inputs, ((0, 0), (pad_beg, pad_end), (pad_beg, pad_end), (0, 0)))


class Conv2DFixedPadding(nn.Module):
    filters: int
    kernel_size: int
    strides: int
    name: str = None

    @nn.compact
    def __call__(self, x):
        if self.strides > 1:
            x = fixed_padding(x, self.kernel_size)
        return nn.Conv(
            self.filters,
            (self.kernel_size, self.kernel_size),
            self.strides,
            padding="SAME" if self.strides == 1 else "VALID",
            use_bias=False,
            name=self.name
        )(x)


class STEM(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = Conv2DFixedPadding(32, kernel_size=3, strides=2, name="stem_conv_1")(x)
        x = nn.BatchNorm(name="stem_bn_1", use_running_average=not train)(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(32, kernel_size=3, strides=1, name="stem_conv_2")(x)
        x = nn.BatchNorm(name="stem_bn_2", use_running_average=not train)(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(64, kernel_size=3, strides=1, name="stem_conv_3")(x)
        x = nn.BatchNorm(name="stem_bn_3", use_running_average=not train)(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(64, kernel_size=3, strides=1, name="stem_conv_4")(x)
        x = nn.BatchNorm(name="stem_bn_4", use_running_average=not train)(x)
        x = nn.relu(x)
        return x


class BlockGroup(nn.Module):
    filters: int
    strides: int
    num_repeats: int
    counter: int
    name: str = None

    @nn.compact
    def __call__(self, x, train=True):
        if self.name is None:
            self.name = f"block_group_{self.counter}"
        x = BottleneckBlock(
            self.filters, strides=self.strides, use_projection=True, name=self.name + "_block_0"
        )(x, train)
        for i in range(1, self.num_repeats):
            x = BottleneckBlock(
                self.filters, strides=1, use_projection=False, name=self.name + f"_block_{i}_"
            )(x, train)
        return x


class BottleneckBlock(nn.Module):
    filters: int
    strides: int
    use_projection: bool
    name: str

    @nn.compact
    def __call__(self, x, train=True):
        shortcut = x
        if self.use_projection:
            filters_out = self.filters * 4
            if self.strides == 2:
                shortcut = nn.avg_pool(x, (2, 2), (2, 2), padding="SAME")
                shortcut = Conv2DFixedPadding(
                    filters_out,
                    kernel_size=1,
                    strides=1,
                    name=f"{self.name}_projection_conv"
                )(shortcut)
            else:
                shortcut = Conv2DFixedPadding(
                    filters_out,
                    kernel_size=1,
                    strides=self.strides,
                    name=f"{self.name}_projection_conv"
                )(shortcut)
            shortcut = nn.BatchNorm(
                axis=3, momentum=0.0, epsilon=1e-5, use_running_average=not train,
                name=f"{self.name}_projection_batch_norm"
            )(shortcut)
        x = Conv2DFixedPadding(self.filters, kernel_size=1, strides=1, name=self.name + "_conv_1")(x)
        x = nn.BatchNorm(axis=3, momentum=0.0, epsilon=1e-5, use_running_average=not train,
            name=f"{self.name}_batch_norm_1"
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(
            self.filters, kernel_size=3, strides=self.strides, name=self.name + "_conv_2"
        )(x)
        x = nn.BatchNorm(axis=3, momentum=0.0, epsilon=1e-5, use_running_average=not train,
            name=f"{self.name}_batch_norm_2"
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(
            self.filters * 4, kernel_size=1, strides=1, name=self.name + "_conv_3"
        )(x)
        x = nn.BatchNorm(axis=3, momentum=0.0, epsilon=1e-5, use_running_average=not train,
            name=f"{self.name}_batch_norm_3"
        )(x)
        x = SE(self.filters, se_ratio=0.25, name=f"{self.name}_se")(x)
        x = x + shortcut
        x = nn.relu(x)
        return x


class SE(nn.Module):
    in_filters: int
    se_ratio: float = 0.25
    expand_ratio: int = 1
    name: str = "se"

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = jnp.mean(x, axis=(-2, -1))  # global average pooling
        se_shape = (x.shape[0], 1, 1, x.shape[-1])
        x = x.reshape(se_shape)
        num_reduced_filters = max(1, int(self.in_filters * 4 * self.se_ratio))
        x = nn.Conv(
            num_reduced_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            name=f"{self.name}_se_reduce"
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            4 * self.in_filters * self.expand_ratio,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            name=f"{self.name}_se_expand"
        )(x)
        x = nn.sigmoid(x)
        return inputs * x


class ResNetRS50(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train=True, representation=False):
        x = STEM()(x, train)
        block_args = [
            { "input_filters": 64, "num_repeats": 3 },
            { "input_filters": 128, "num_repeats": 4 },
            { "input_filters": 256, "num_repeats": 6 },
            { "input_filters": 512, "num_repeats": 3 },
        ]
        for i, block_arg in enumerate(block_args):
            x = BlockGroup(
                block_arg["input_filters"],
                strides=(1 if i == 0 else 2),
                num_repeats=block_arg["num_repeats"],
                counter=i
            )(x, train)
        # global average pooling
        x = jnp.mean(x, axis=(-2, -1))
        x = nn.Dropout(0.25, deterministic=train, name="top_dropout")(x)
        if representation:
            return x
        x = nn.Dense(self.num_classes, name="classifier")(x)
        x = nn.softmax(x)
        return x
