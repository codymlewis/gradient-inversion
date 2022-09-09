import jax.numpy as jnp
import flax.linen as nn
import einops


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(x, kernel_size):
    input_size = x.shape[1:3]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((0, 0), (correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]), (0, 0))


class InvertedResBlock(nn.Module):
    filters: int
    alpha: float
    stride: int
    expansion: int
    block_id: int

    @nn.compact
    def __call__(self, x, train=True):
        inputs = x
        in_channels = x.shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        
        if self.block_id:
            prefix = f"block_{self.block_id}_"
            x = nn.Conv(
                self.expansion * in_channels,
                (1, 1),
                padding='SAME',
                use_bias=False,
                name=prefix + "expand"
            )(x)
            x = nn.BatchNorm(
                use_running_average=not train,
                axis=-1,
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand_BN'
            )(x)
            x = nn.relu6(x)
        else:
            prefix = 'expanded_conv_'

        if self.stride == 2:
            x = jnp.pad(x, correct_pad(x, 3))
        x = nn.Conv(
            x.shape[-1], (3, 3), feature_group_count=x.shape[-1], strides=self.stride, use_bias=False,
            padding="SAME" if self.stride == 1 else "VALID",
            name=prefix + "depthwise"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train, axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + "depthwise_BN"
        )(x)
        x = nn.relu6(x)
        x = nn.Conv(
            pointwise_filters, (1, 1), padding='SAME', use_bias=True,
            name=prefix + 'project'
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train, axis=-1, epsilon=1e-3, momentum=0.999,
            name=prefix + "project_BN"
        )(x)
        if in_channels == pointwise_filters and self.stride == 1:
            return inputs + x
        return x


class MobileNetV2(nn.Module):
    classes: int = 1000
    alpha: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        first_block_filters = _make_divisible(32 * self.alpha, 8)
        x = nn.Conv(
            first_block_filters, (3, 3), strides=(2, 2), padding="SAME", use_bias=False, name="Conv1"
        )(x)
        x = nn.BatchNorm(use_running_average=not train, axis=-1, epsilon=1e-3, momentum=0.999, name="bn_Conv1")(x)
        x = nn.relu6(x)
        x = InvertedResBlock(filters=16, alpha=self.alpha, stride=1, expansion=1, block_id=0)(x, train)
        x = InvertedResBlock(filters=24, alpha=self.alpha, stride=2, expansion=6, block_id=1)(x, train)
        x = InvertedResBlock(filters=24, alpha=self.alpha, stride=1, expansion=6, block_id=2)(x, train)

        x = InvertedResBlock(filters=32, alpha=self.alpha, stride=2, expansion=6, block_id=3)(x, train)
        x = InvertedResBlock(filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=4)(x, train)
        x = InvertedResBlock(filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=5)(x, train)

        x = InvertedResBlock(filters=64, alpha=self.alpha, stride=2, expansion=6, block_id=6)(x, train)
        x = InvertedResBlock(filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=7)(x, train)
        x = InvertedResBlock(filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=8)(x, train)
        x = InvertedResBlock(filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=9)(x, train)

        x = InvertedResBlock(filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=10)(x, train)
        x = InvertedResBlock(filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=11)(x, train)
        x = InvertedResBlock(filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=12)(x, train)

        x = InvertedResBlock(filters=160, alpha=self.alpha, stride=2, expansion=6, block_id=13)(x, train)
        x = InvertedResBlock(filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=14)(x, train)
        x = InvertedResBlock(filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=15)(x, train)

        x = InvertedResBlock(filters=320, alpha=self.alpha, stride=1, expansion=6, block_id=16)(x, train)

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we increase the number of output channels
        if self.alpha > 1.0:
            last_block_filters = _make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280

        x = nn.Conv(last_block_filters, (1, 1), padding='VALID', use_bias=False, name="Conv_1")(x)
        x = nn.BatchNorm(
            use_running_average=not train, axis=-1, epsilon=1e-3, momentum=0.999, name="Conv_1_bn"
        )(x)
        x = nn.relu6(x)
        x = einops.reduce(x, "b h w d -> b d", 'mean')  # Global average pooling
        x = nn.Dense(self.classes, name="predictions")(x)
        x = nn.softmax(x)
        return x
