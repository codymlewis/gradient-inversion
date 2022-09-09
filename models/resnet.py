from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp
import einops


class ResNet(nn.Module):
    stack_fn: Callable[[jnp.array, bool], jnp.array]
    preact: bool
    use_bias: bool
    model_name: str = "resnet"
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        x = jnp.pad(x, ((0, 0), (3, 3), (3, 3), (0, 0)))
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="VALID", use_bias=self.use_bias, name="conv1_conv")(x)

        if not self.preact:
            x = nn.BatchNorm(use_running_average=not train, axis=3, epsilon=1.001e-5, name="conv1_bn")(x)
            x = nn.relu(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = self.stack_fn(x, train)

        if self.preact:
            x = nn.BatchNorm(use_running_average=not train, axis=3, epsilon=1.001e-5, name="post_bn")(x)
            x = nn.relu(x)

        x = einops.reduce(x, "b h w d -> b d", 'mean')
        x = nn.Dense(self.classes, name="predictions")(x)
        x = nn.softmax(x)
        return x


class Block1(nn.Module):
    filters: int
    kernel: (int, int) = (3, 3)
    strides: (int, int) = (1, 1)
    conv_shortcut: bool = True
    name: str = None

    @nn.compact
    def __call__(self, x, train=True):
        if self.conv_shortcut:
            shortcut = nn.Conv(
                4 * self.filters, (1, 1), strides=self.strides, padding="VALID", name=self.name + "_0_conv"
            )(x)
            shortcut = nn.BatchNorm(
                use_running_average=not train, axis=3, epsilon=1.001e-5, name=self.name + "_0_bn"
            )(shortcut)
        else:
            shortcut = x
        
        x = nn.Conv(self.filters, (1, 1), strides=self.strides, padding="VALID", name=self.name + "_1_conv")(x)
        x = nn.BatchNorm(
            use_running_average=not train, axis=3, epsilon=1.001e-5, name=self.name + "_1_bn"
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            self.filters, self.kernel, padding="SAME", name=self.name + "_2_conv"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train, axis=3, epsilon=1.001e-5, name=self.name + "_2_bn"
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            4 * self.filters, (1, 1), padding="VALID", name=self.name + "_3_conv"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train, axis=3, epsilon=1.001e-5, name=self.name + "_3_bn"
        )(x)

        x = shortcut + x
        x = nn.relu(x)
        return x


class Stack1(nn.Module):
    filters: int
    blocks: int
    strides1: (int, int) = (2, 2)
    name: str = None

    @nn.compact
    def __call__(self, x, train=True):
        x = Block1(self.filters, strides=self.strides1, name=self.name + "_block1")(x, train)
        for i in range(2, self.blocks + 1):
            x = Block1(self.filters, conv_shortcut=False, name=f"{self.name}_block{i}")(x, train)
        return x


class Block2(nn.Module):
    filters: int
    kernel: (int, int) = (3, 3)
    strides: (int, int) = (1, 1)
    conv_shortcut: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x, train=True):
        preact = nn.BatchNorm(
            use_running_average=not train, axis=3, epsilon=1.001e-5, name=self.name + "_preact_bn"
        )(x)
        preact = nn.relu(preact)

        if self.conv_shortcut:
            shortcut = nn.Conv(
                4 * self.filters, (1, 1), strides=self.strides, padding="VALID", name=self.name + "_0_conv"
            )(preact)
        else:
            shortcut = nn.max_pool(x, (1, 1), strides=self.strides) if self.strides > (1, 1) else x

        x = nn.Conv(
            self.filters, (1, 1), strides=(1, 1), padding="VALID", use_bias=False,
            name=self.name + "_1_conv"
        )(preact)
        x = nn.BatchNorm(use_running_average=not train, axis=3, epsilon=1.001e-5, name=self.name + "_1_bn")(x)
        x = nn.relu(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.Conv(
            self.filters, self.kernel, strides=self.strides, padding="VALID", use_bias=False,
            name=self.name + "_2_conv"
        )(x)
        x = nn.BatchNorm(use_running_average=not train, axis=3, epsilon=1.001e-5, name=self.name + "_2_bn")(x)
        x = nn.relu(x)

        x = nn.Conv(4 * self.filters, (1, 1), name=self.name + "_3_conv")(x)
        x = shortcut + x
        return x


class Stack2(nn.Module):
    filters: int
    blocks: int
    strides1: (int, int) = (2, 2)
    name: str = None

    @nn.compact
    def __call__(self, x, train=True):
        x = Block2(self.filters, conv_shortcut=True, name=self.name + "_block1")(x, train)
        for i in range(2, self.blocks):
            x = Block2(self.filters, name=f"{self.name}_block{i}")(x, train)
        x = Block2(self.filters, strides=self.strides1, name=f"{self.name}_block{self.blocks}")(x, train)
        return x


class ResNet50(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        def stack_fn(x, train=True):
            x = Stack1(64, 3, strides1=(1, 1), name="conv2")(x, train)
            x = Stack1(128, 4, name="conv3")(x, train)
            x = Stack1(256, 6, name="conv4")(x, train)
            return Stack1(512, 3, name="conv5")(x, train)

        return ResNet(stack_fn, False, True, "resnet50", classes=self.classes)(x, train)


class ResNet101(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        def stack_fn(x, train=True):
            x = Stack1(64, 3, strides1=(1, 1), name="conv2")(x, train)
            x = Stack1(128, 4, name="conv3")(x, train)
            x = Stack1(256, 23, name="conv4")(x, train)
            return Stack1(512, 3, name="conv5")(x, train)

        return ResNet(stack_fn, False, True, "resnet101", classes=self.classes)(x, train)


class ResNet152(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        def stack_fn(x, train=True):
            x = Stack1(64, 3, strides1=(1, 1), name="conv2")(x, train)
            x = Stack1(128, 8, name="conv3")(x, train)
            x = Stack1(256, 36, name="conv4")(x, train)
            return Stack1(512, 3, name="conv5")(x, train)

        return ResNet(stack_fn, False, True, "resnet152", classes=self.classes)(x, train)
