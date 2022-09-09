"""
ResNetRS implementations adapted from keras.applications
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import einops

# Note: The batch norms require specialized training functions, checkout the resnetrs branch

BLOCK_ARGS = {
    50: [
        {
            "input_filters": 64,
            "num_repeats": 3
        },
        {
            "input_filters": 128,
            "num_repeats": 4
        },
        {
            "input_filters": 256,
            "num_repeats": 6
        },
        {
            "input_filters": 512,
            "num_repeats": 3
        },
    ],
    101: [
        {
            "input_filters": 64,
            "num_repeats": 3
        },
        {
            "input_filters": 128,
            "num_repeats": 4
        },
        {
            "input_filters": 256,
            "num_repeats": 23
        },
        {
            "input_filters": 512,
            "num_repeats": 3
        },
    ],
    152: [
        {
            "input_filters": 64,
            "num_repeats": 3
        },
        {
            "input_filters": 128,
            "num_repeats": 8
        },
        {
            "input_filters": 256,
            "num_repeats": 36
        },
        {
            "input_filters": 512,
            "num_repeats": 3
        },
    ],
    200: [
        {
            "input_filters": 64,
            "num_repeats": 3
        },
        {
            "input_filters": 128,
            "num_repeats": 24
        },
        {
            "input_filters": 256,
            "num_repeats": 36
        },
        {
            "input_filters": 512,
            "num_repeats": 3
        },
    ],
    270: [
        {
            "input_filters": 64,
            "num_repeats": 4
        },
        {
            "input_filters": 128,
            "num_repeats": 29
        },
        {
            "input_filters": 256,
            "num_repeats": 53
        },
        {
            "input_filters": 512,
            "num_repeats": 4
        },
    ],
    350: [
        {
            "input_filters": 64,
            "num_repeats": 4
        },
        {
            "input_filters": 128,
            "num_repeats": 36
        },
        {
            "input_filters": 256,
            "num_repeats": 72
        },
        {
            "input_filters": 512,
            "num_repeats": 4
        },
    ],
    420: [
        {
            "input_filters": 64,
            "num_repeats": 4
        },
        {
            "input_filters": 128,
            "num_repeats": 44
        },
        {
            "input_filters": 256,
            "num_repeats": 87
        },
        {
            "input_filters": 512,
            "num_repeats": 4
        },
    ],
}


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
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=self.name
        )(x)


class STEM(nn.Module):
    bn_momentum: float = 0.0
    bn_epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, train=True):
        x = Conv2DFixedPadding(32, kernel_size=3, strides=2, name="stem_conv_1")(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_1",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(32, kernel_size=3, strides=1, name="stem_conv_2")(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_2",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(64, kernel_size=3, strides=1, name="stem_conv_3")(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_3",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(64, kernel_size=3, strides=1, name="stem_conv_4")(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_4",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        return x


class BlockGroup(nn.Module):
    filters: int
    strides: int
    num_repeats: int
    counter: int
    se_ratio: float = 0.25
    bn_epsilon: float = 1e-5
    bn_momentum: float = 0.0
    survival_probability: float = 0.8
    name: str = None

    @nn.compact
    def __call__(self, x, train=True):
        if self.name is None:
            self.name = f"block_group_{self.counter}"
        x = BottleneckBlock(
            self.filters,
            strides=self.strides,
            use_projection=True,
            se_ratio=self.se_ratio,
            bn_epsilon=self.bn_epsilon,
            bn_momentum=self.bn_momentum,
            survival_probability=self.survival_probability,
            name=self.name + "_block_0"
        )(x, train)
        for i in range(1, self.num_repeats):
            x = BottleneckBlock(
                self.filters,
                strides=1,
                use_projection=False,
                se_ratio=self.se_ratio,
                bn_epsilon=self.bn_epsilon,
                bn_momentum=self.bn_momentum,
                survival_probability=self.survival_probability,
                name=self.name + f"_block_{i}_"
            )(x, train)
        return x


class BottleneckBlock(nn.Module):
    filters: int
    strides: int
    use_projection: bool
    bn_momentum: float = 0.0
    bn_epsilon: float = 1e-5
    survival_probability: float = 0.8
    se_ratio: float = 0.25
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
                axis=3, momentum=self.bn_momentum, epsilon=self.bn_epsilon, use_running_average=not train,
                name=f"{self.name}_projection_batch_norm"
            )(shortcut)
        x = Conv2DFixedPadding(self.filters, kernel_size=1, strides=1, name=self.name + "_conv_1")(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon, 
            use_running_average=not train,
            name=f"{self.name}_batch_norm_1"
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(
            self.filters, kernel_size=3, strides=self.strides, name=self.name + "_conv_2"
        )(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train,
            name=f"{self.name}_batch_norm_2"
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(
            self.filters * 4, kernel_size=1, strides=1, name=self.name + "_conv_3"
        )(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train,
            name=f"{self.name}_batch_norm_3"
        )(x)
        if 0 < self.se_ratio < 1:
            x = SE(self.filters, se_ratio=self.se_ratio, name=f"{self.name}_se")(x)
        if self.survival_probability:
            x = nn.Dropout(self.survival_probability, deterministic=train, name=f"{self.name}_drop")(x)
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
        x = einops.reduce(x, 'b h w d -> b 1 1 d', 'mean')  # global average pooling
        num_reduced_filters = max(1, int(self.in_filters * 4 * self.se_ratio))
        x = nn.Conv(
            num_reduced_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=f"{self.name}_se_reduce"
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            4 * self.in_filters * self.expand_ratio,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=f"{self.name}_se_expand"
        )(x)
        x = nn.sigmoid(x)
        return inputs * x


class ResNetRS(nn.Module):
    classes: int
    block_args: dict
    drop_connect_rate: float = 0.2
    dropout_rate: float = 0.25
    bn_momentum: float = 0.0
    bn_epsilon: float = 1e-5
    se_ratio: float = 0.25

    @nn.compact
    def __call__(self, x, train=True):
        x = STEM(
            bn_momentum=self.bn_momentum, bn_epsilon=self.bn_epsilon, name="STEM_1"
        )(x, train)
        for i, block_arg in enumerate(self.block_args):
            survival_probability = self.drop_connect_rate * float(i + 2) / (len(self.block_args) + 1)
            x = BlockGroup(
                block_arg["input_filters"],
                strides=(1 if i == 0 else 2),
                num_repeats=block_arg["num_repeats"],
                counter=i,
                se_ratio=self.se_ratio,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon,
                survival_probability=survival_probability,
                name=f"BlockGroup{i + 2}"
            )(x, train) 
        x = einops.reduce(x, 'b h w d -> b d', 'mean')  # global average pooling
        x = nn.Dropout(self.dropout_rate, deterministic=train, name="top_dropout")(x)
        x = nn.Dense(self.classes, name="predictions")(x)
        x = nn.softmax(x)
        return x


class ResNetRS50(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return ResNetRS(
            self.classes,
            BLOCK_ARGS[50],
            drop_connect_rate=0.0,
            dropout_rate=0.25,
        )(x, train)


class ResNetRS101(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return ResNetRS(
            self.classes,
            BLOCK_ARGS[101],
            drop_connect_rate=0.0,
            dropout_rate=0.25,
        )(x, train)


class ResNetRS152(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return ResNetRS(
            self.classes,
            BLOCK_ARGS[152],
            drop_connect_rate=0.0,
            dropout_rate=0.25,
        )(x, train)


class ResNetRS200(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return ResNetRS(
            self.classes,
            BLOCK_ARGS[200],
            drop_connect_rate=0.1,
            dropout_rate=0.25,
        )(x, train)


class ResNetRS270(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return ResNetRS(
            self.classes,
            BLOCK_ARGS[270],
            drop_connect_rate=0.1,
            dropout_rate=0.25,
        )(x, train)


class ResNetRS350(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return ResNetRS(
            self.classes,
            BLOCK_ARGS[350],
            drop_connect_rate=0.1,
            dropout_rate=0.4,
        )(x, train)


class ResNetRS420(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return ResNetRS(
            self.classes,
            BLOCK_ARGS[420],
            drop_connect_rate=0.1,
            dropout_rate=0.4,
        )(x, train)
