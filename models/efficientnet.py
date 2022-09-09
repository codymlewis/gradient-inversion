import copy
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import einops


DEFAULT_BLOCKS_ARGS = [{
    'kernel': (3, 3),
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel': (3, 3),
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel': (5, 5),
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel': (3, 3),
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel': (5, 5),
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel': (5, 5),
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel': (3, 3),
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]


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



class Block(nn.Module):
    drop_rate: float = 0.
    filters_in: int = 32
    filters_out: int = 16
    kernel: (int, int) = (3, 3)
    strides: int = 1
    expand_ratio: float = 1
    se_ratio: float = 0.
    id_skip: bool = True
    name: str = ''

    @nn.compact
    def __call__(self, x, train=True):
        inputs = x
        # Expansion phase
        filters = self.filters_in * self.expand_ratio
        if self.expand_ratio != 1:
            x = nn.Conv(filters, (1, 1), padding="SAME", use_bias=False, name=self.name + "expand_conv")(x)
            x = nn.BatchNorm(use_running_average=not train, axis=3, name=self.name + "expand_bn")(x)
            x = nn.swish(x)
        
        # Depthwise convolution
        if self.strides == 2:
            x = jnp.pad(x, correct_pad(x, self.kernel[0]))
            conv_pad = 'VALID'
        else:
            conv_pad = 'SAME'
        x = nn.Conv(
            x.shape[-1], self.kernel, feature_group_count=x.shape[-1], strides=self.strides, use_bias=False,
            padding=conv_pad, name=self.name + "dwconv"
        )(x)
        x = nn.BatchNorm(use_running_average=not train, axis=3, name=self.name + "bn")(x)
        x = nn.swish(x)

        # Squeeze and Excitation phase
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.filters_in * self.se_ratio))
            se = einops.reduce(x, 'b h w d -> b d', 'mean')
            se = jnp.reshape(se, (-1, 1, 1, filters))
            se = nn.Conv(filters_se, (1, 1), padding="SAME", name=self.name + "se_reduce")(se)
            se = nn.swish(se)
            se = nn.Conv(filters, (1, 1), padding="SAME", name=self.name + "se_expand")(se)
            x = x * se

        # Output phase
        x = nn.Conv(self.filters_out, (1, 1), padding="SAME", use_bias=False, name=self.name + "project_conv")(x)
        x = nn.BatchNorm(use_running_average=not train, axis=3, name=self.name + "project_bn")(x)
        if self.id_skip and self.strides == 1 and self.filters_in == self.filters_out:
            if self.drop_rate > 0:
                x = nn.Dropout(self.drop_rate, deterministic=train, name=self.name + "drop")(x)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    width_coefficient: float
    depth_coefficient: float
    default_size: float
    dropout_rate: float = 0.2
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    block_args: str|dict = 'default'
    model_name: str = 'efficientnet'
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        if self.block_args == "default":
            block_args = copy.deepcopy(DEFAULT_BLOCKS_ARGS)
        else:
            block_args = copy.deepcopy(self.block_args)

        def round_filters(filters, divisor=self.depth_divisor):
            filters *= self.width_coefficient
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return  int(new_filters)

        def round_repeats(repeats):
            return int(math.ceil(self.depth_coefficient * repeats))

        # To match the original implementation where normalization uses var instead of std
        x = x / jnp.sqrt(jnp.array([0.229, 0.224, 0.225]))

        x = jnp.pad(x, correct_pad(x, 3))
        x = nn.Conv(
            round_filters(32), (3, 3), strides=(2, 2), padding="VALID", use_bias=False, name='stem_conv'
        )(x)
        x = nn.BatchNorm(use_running_average=not train, axis=3, name='stem_bn')(x)
        x = nn.swish(x)

        b = 0
        blocks = float(sum(round_repeats(args['repeats']) for args in block_args))
        for (i, args) in enumerate(block_args):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):
                # The first block needs to take care of stride and filter size increase
                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = Block(
                    self.drop_connect_rate * b / blocks,
                    name=f"block{i + 1}{chr(j + 97)}_",
                    **args
                )(x, train)
                b += 1

        # Build top
        x = nn.Conv(round_filters(1280), (1, 1), padding="SAME", use_bias=False, name='top_conv')(x)
        x = nn.BatchNorm(use_running_average=not train, axis=3, name='top_bn')(x)
        x = nn.swish(x)
        x = einops.reduce(x, 'b h w d -> b d', 'mean')
        if self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate, deterministic=train, name='top_dropout')(x)
        x = nn.Dense(self.classes, name='predictions')(x)
        x = nn.softmax(x)
        return x
    

class EfficientNetB0(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return EfficientNet(1.0, 1.0, 224, 0.2, model_name='efficientnetb0', classes=self.classes)(x, train)


class EfficientNetB1(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return EfficientNet(1.0, 1.1, 240, 0.2, model_name='efficientnetb1', classes=self.classes)(x, train)


class EfficientNetB2(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return EfficientNet(1.1, 1.2, 260, 0.3, model_name='efficientnetb2', classes=self.classes)(x, train)


class EfficientNetB3(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return EfficientNet(1.2, 1.4, 300, 0.3, model_name='efficientnetb3', classes=self.classes)(x, train)


class EfficientNetB4(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return EfficientNet(1.4, 1.8, 380, 0.4, model_name='efficientnetb4', classes=self.classes)(x, train)


class EfficientNetB5(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return EfficientNet(1.6, 2.2, 456, 0.4, model_name='efficientnetb5', classes=self.classes)(x, train)


class EfficientNetB6(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return EfficientNet(1.8, 2.6, 528, 0.5, model_name='efficientnetb6', classes=self.classes)(x, train)


class EfficientNetB7(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        return EfficientNet(2.0, 3.1, 600, 0.5, model_name='efficientnetb7', classes=self.classes)(x, train)
