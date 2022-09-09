import jax.numpy as jnp
import flax.linen as nn
import einops


class ConvBlock(nn.Module):
    growth_rate: int
    name: str

    @nn.compact
    def __call__(self, x, train=True):
        x1 = nn.BatchNorm(axis=3, epsilon=1.001e-5, name=self.name + '_0_bn', use_running_average=not train)(x)
        x1 = nn.relu(x1)
        x1 = nn.Conv(4 * self.growth_rate, (1, 1), padding='VALID', use_bias=False, name=self.name + '_1_conv')(x1)
        x1 = nn.BatchNorm(axis=3, epsilon=1.001e-5, name=self.name + '_1_bn', use_running_average=not train)(x1)
        x1 = nn.relu(x1)
        x1 = nn.Conv(self.growth_rate, (3, 3), padding='SAME', use_bias=False, name=self.name + '_2_conv')(x1)
        x = jnp.concatenate((x, x1), axis=3)
        return x


class TransitionBlock(nn.Module):
    reduction: float
    name: str

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.BatchNorm(axis=3, epsilon=1.001e-5, name=self.name + '_bn', use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(int(x.shape[3] * self.reduction), (1, 1), padding='VALID', use_bias=False, name=self.name + '_conv')(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x


class DenseBlock(nn.Module):
    blocks: list[int]
    name: str

    @nn.compact
    def __call__(self, x, train=True):
        for i in range(self.blocks):
            x = ConvBlock(32, name=f"{self.name}_block{i + 1}")(x, train)
        return x


class DenseNet(nn.Module):
    classes: int
    blocks: list[int]

    @nn.compact
    def __call__(self, x, train=True, representation=False):
        x = jnp.pad(x, ((0, 0), (3, 3), (3, 3), (0, 0)))
        x = nn.Conv(64, (7, 7), (2, 2), padding='VALID', use_bias=False, name="conv1/conv")(x)
        x = nn.BatchNorm(axis=3, epsilon=1.001e-5, name='conv1/bn', use_running_average=not train)(x)
        x = nn.relu(x)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.max_pool(x, (3, 3), (2, 2))
        x = DenseBlock(self.blocks[0], name="conv2")(x, train)
        x = TransitionBlock(0.5, name="pool2")(x, train)
        x = DenseBlock(self.blocks[1], name="conv3")(x, train)
        x = TransitionBlock(0.5, name="pool3")(x, train)
        x = DenseBlock(self.blocks[2], name="conv4")(x, train)
        x = TransitionBlock(0.5, name="pool4")(x, train)
        x = DenseBlock(self.blocks[3], name="conv5")(x, train)
        x = nn.BatchNorm(axis=3, epsilon=1.001e-5, name="bn", use_running_average=not train)(x)
        x = nn.relu(x)
        x = einops.reduce(x, "b w h d -> b d", "mean")  # Global average pooling
        if representation:
            return x
        x = nn.Dense(self.classes, name="predictions")(x)
        x = nn.softmax(x)
        return x


class DenseNet121(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True, representation=False):
        return DenseNet(self.classes, [6, 12, 24, 16])(x, train, representation)


class DenseNet169(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True, representation=False):
        return DenseNet(self.classes, [6, 12, 32, 32])(x, train, representation)


class DenseNet201(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True, representation=False):
        return DenseNet(self.classes, [6, 12, 48, 32])(x, train, representation)
