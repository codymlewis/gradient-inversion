import flax.linen as nn

from . import resnet


class ResNet50V2(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        def stack_fn(x, train=True):
            x = resnet.Stack2(64, 3, name="conv2")(x, train)
            x = resnet.Stack2(128, 4, name="conv3")(x, train)
            x = resnet.Stack2(256, 6, name="conv4")(x, train)
            return resnet.Stack2(512, 3, strides1=(1, 1), name="conv5")(x, train)

        return resnet.ResNet(stack_fn, True, True, "resnet50v2", classes=self.classes)(x, train)


class ResNet101V2(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        def stack_fn(x, train=True):
            x = resnet.Stack2(64, 3, name="conv2")(x, train)
            x = resnet.Stack2(128, 4, name="conv3")(x, train)
            x = resnet.Stack2(256, 23, name="conv4")(x, train)
            return resnet.Stack2(512, 3, strides1=(1, 1), name="conv5")(x, train)

        return resnet.ResNet(stack_fn, True, True, "resnet101v2", classes=self.classes)(x, train)


class ResNet152V2(nn.Module):
    classes: int = 1000

    @nn.compact
    def __call__(self, x, train=True):
        def stack_fn(x, train=True):
            x = resnet.Stack2(64, 3, name="conv2")(x, train)
            x = resnet.Stack2(128, 8, name="conv3")(x, train)
            x = resnet.Stack2(256, 36, name="conv4")(x, train)
            return resnet.Stack2(512, 3, strides1=(1, 1), name="conv5")(x, train)

        return resnet.ResNet(stack_fn, True, True, "resnet152v2", classes=self.classes)(x, train)
