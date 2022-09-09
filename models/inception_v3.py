import jax.numpy as jnp
import flax.linen as nn
import einops


class ConvBN(nn.Module):
    filters: int
    kernel: tuple[int]
    padding: str = 'SAME'
    strides: tuple[int] = (1, 1)
    name: str = None

    @nn.compact
    def __call__(self, x, train=True):
        if self.name is None:
            bn_name = self.name + '_bn'
            conv_name = self.name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = nn.Conv(
            self.filters,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
            name=conv_name
        )(x)
        x = nn.BatchNorm(axis=3, name=bn_name)(
            x, use_running_average=not train, use_scale=False
        )
        x = nn.relu(x)
        return x


class InceptionV3(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        x = ConvBN(32, (3, 3), strides=(2, 2), padding="VALID")(x, train)
        x = ConvBN(32, (3, 3), padding='VALID')(x, train)
        x = ConvBN(64, (3, 3))(x, train)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = ConvBN(80, (1, 1), padding='VALID')(x, train)
        x = ConvBN(192, (3, 3), padding='VALID')(x, train)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        # mixed 0: 35 x 35 x 256
        branch1x1 = ConvBN(64, (1, 1))(x, train)

        branch5x5 = ConvBN(48, (1, 1))(x, train)
        branch5x5 = ConvBN(64, (5, 5))(branch5x5, train)

        branch3x3dbl = ConvBN(64, (1, 1))(x, train)
        branch3x3dbl = ConvBN(96, (3, 3))(branch3x3dbl, train)
        branch3x3dbl = ConvBN(96, (3, 3))(branch3x3dbl, train)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvBN(32, (1, 1))(branch_pool, train)
        x = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=3)

        # mixed 1: 35 x 35 x 288
        branch1x1 = ConvBN(64, (1, 1))(x, train)

        branch5x5 = ConvBN(48, (1, 1))(x, train)
        branch5x5 = ConvBN(64, (5, 5))(branch5x5, train)

        branch3x3dbl = ConvBN(64, (1, 1))(x, train)
        branch3x3dbl = ConvBN(96, (3, 3))(branch3x3dbl, train)
        branch3x3dbl = ConvBN(96, (3, 3))(branch3x3dbl, train)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvBN(64, (1, 1))(branch_pool, train)
        x = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=3)

        # mixed 2: 35 x 35 x 288
        branch1x1 = ConvBN(64, (1, 1))(x, train)

        branch5x5 = ConvBN(48, (1, 1))(x, train)
        branch5x5 = ConvBN(64, (5, 5))(branch5x5, train)

        branch3x3dbl = ConvBN(64, (1, 1))(x, train)
        branch3x3dbl = ConvBN(96, (3, 3))(branch3x3dbl, train)
        branch3x3dbl = ConvBN(96, (3, 3))(branch3x3dbl, train)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvBN(64, (1, 1))(branch_pool, train)
        x = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=3)

        # mixed 3: 17 x 17 x 768
        branch3x3 = ConvBN(384, (3, 3), strides=(2, 2), padding='VALID')(x, train)

        branch3x3dbl = ConvBN(64, (1, 1))(x, train)
        branch3x3dbl = ConvBN(96, (3, 3))(branch3x3dbl, train)
        branch3x3dbl = ConvBN(96, (3, 3), strides=(2, 2), padding='VALID')(branch3x3dbl, train)

        branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2))
        x = jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=3)

        # mixed 4: 17 x 17 x 768
        branch1x1 = ConvBN(192, (1, 1))(x, train)

        branch7x7 = ConvBN(128, (1, 1))(x, train)
        branch7x7 = ConvBN(128, (1, 7))(branch7x7, train)
        branch7x7 = ConvBN(192, (7, 1))(branch7x7, train)

        branch7x7dbl = ConvBN(128, (1, 1))(x, train)
        branch7x7dbl = ConvBN(128, (7, 1))(branch7x7dbl, train)
        branch7x7dbl = ConvBN(128, (1, 7))(branch7x7dbl, train)
        branch7x7dbl = ConvBN(128, (7, 1))(branch7x7dbl, train)
        branch7x7dbl = ConvBN(192, (1, 7))(branch7x7dbl, train)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvBN(192, (1, 1))(branch_pool, train)
        x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=3)

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = ConvBN(192, (1, 1))(x, train)

            branch7x7 = ConvBN(160, (1, 1))(x, train)
            branch7x7 = ConvBN(160, (1, 7))(branch7x7, train)
            branch7x7 = ConvBN(192, (7, 1))(branch7x7, train)

            branch7x7dbl = ConvBN(160, (1, 1))(x, train)
            branch7x7dbl = ConvBN(160, (7, 1))(branch7x7dbl, train)
            branch7x7dbl = ConvBN(160, (1, 7))(branch7x7dbl, train)
            branch7x7dbl = ConvBN(160, (7, 1))(branch7x7dbl, train)
            branch7x7dbl = ConvBN(192, (1, 7))(branch7x7dbl, train)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = ConvBN(192, (1, 1))(branch_pool, train)
            x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=3)

        # mixed 7: 17 x 17 x 768
        branch1x1 = ConvBN(192, (1, 1))(x, train)

        branch7x7 = ConvBN(192, (1, 1))(x, train)
        branch7x7 = ConvBN(192, (1, 7))(branch7x7, train)
        branch7x7 = ConvBN(192, (7, 1))(branch7x7, train)

        branch7x7dbl = ConvBN(192, (1, 1))(x, train)
        branch7x7dbl = ConvBN(192, (7, 1))(branch7x7dbl, train)
        branch7x7dbl = ConvBN(192, (1, 7))(branch7x7dbl, train)
        branch7x7dbl = ConvBN(192, (7, 1))(branch7x7dbl, train)
        branch7x7dbl = ConvBN(192, (1, 7))(branch7x7dbl, train)

        branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        branch_pool = ConvBN(192, (1, 1))(branch_pool, train)
        x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=3)

        # mixed 8: 8 x 8 x 1280
        branch3x3 = ConvBN(192, (1, 1))(x, train)
        branch3x3 = ConvBN(320, (3, 3), strides=(2, 2), padding='VALID')(branch3x3, train)

        branch7x7x3 = ConvBN(192, (1, 1))(x, train)
        branch7x7x3 = ConvBN(192, (1, 7))(branch7x7x3, train)
        branch7x7x3 = ConvBN(192, (7, 1))(branch7x7x3, train)
        branch7x7x3 = ConvBN(192, (3, 3), strides=(2, 2), padding='VALID')(branch7x7x3, train)

        branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2))
        x = jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=3)

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = ConvBN(320, (1, 1))(x, train)

            branch3x3 = ConvBN(384, (1, 1))(x, train)
            branch3x3_1 = ConvBN(384, (1, 3))(branch3x3, train)
            branch3x3_2 = ConvBN(384, (3, 1))(branch3x3, train)
            branch3x3 = jnp.concatenate((branch3x3_1, branch3x3_2), axis=3)

            branch3x3dbl = ConvBN(448, (1, 1))(x, train)
            branch3x3dbl = ConvBN(384, (3, 3))(branch3x3dbl, train)
            branch3x3dbl_1 = ConvBN(384, (1, 3))(branch3x3dbl, train)
            branch3x3dbl_2 = ConvBN(384, (3, 1))(branch3x3dbl, train)
            branch3x3dbl = jnp.concatenate((branch3x3dbl_1, branch3x3dbl_2), axis=3)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = ConvBN(192, (1, 1))(branch_pool, train)
            x = jnp.concatenate((branch1x1, branch3x3, branch3x3dbl, branch_pool), axis=3)

        x = einops.reduce(x, 'b w h d -> b d', 'mean')
        x = nn.Dense(self.classes, name='predictions')(x)
        x = nn.softmax(x)
        return x
