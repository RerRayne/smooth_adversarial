'''Pre-activation ResNet in JAX.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import flax
from flax import nn
import jax.numpy as jnp
import jax.experimental.host_callback

import jax.nn


class PreActBlock(nn.Module):
    def apply(self, x, in_planes, planes, stride=1, expansion=1, train=True):
        x = x * 0.0 + 1
        x = jax.experimental.host_callback.id_print(jnp.mean(x), result = x, a = "X mean")
        y = nn.BatchNorm(x,
                         # axis=(1,2,3),
                         use_running_average=not train,
                         # use_running_average=True,
                         momentum=0.9,
                         # momentum=0.2,
                         epsilon=1e-5)
        y = x
        y = jax.experimental.host_callback.id_print(jnp.mean(y), result = y, a = "After batchnorm")

        print(y.shape)
        # y = x
        y = jax.nn.relu(y)
        if stride != 1 or in_planes != expansion * planes:
            short_y = nn.Conv(inputs=y,
                              features=planes * expansion,
                              kernel_size=(1, 1),
                              strides=(stride, stride),
                              padding=((0, 0), (0, 0)),
                              bias=False)
        else:
            short_y = x

        y = nn.Conv(inputs=y,
                    features=planes,
                    kernel_size=(3, 3),
                    strides=(stride, stride),
                    padding=((1, 1), (1, 1)),
                    bias=False)
        z = y
        y = nn.BatchNorm(y,
                         use_running_average=not train,
                         momentum=0.9,
                         epsilon=1e-5)
        y = z
        y = jax.nn.relu(y)
        y = nn.Conv(inputs=y,
                    features=planes,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=((1, 1), (1, 1)),
                    bias=False)
        y += short_y
        # y *= 0.0
        return y


class PreActBottleneck(nn.Module):
    def apply(self, x, in_planes, planes, stride=1, expansion=4, train=True):
        y = nn.BatchNorm(x,
                         use_running_average=not train,
                         momentum=0.9,
                         epsilon=1e-5)
        y = jax.nn.relu(y)
        if stride != 1 or in_planes != expansion * planes:
            short_y = nn.Conv(inputs=y,
                              features=planes * expansion,
                              kernel_size=(1, 1),
                              strides=(stride, stride),
                              padding=((0, 0), (0, 0)),
                              bias=False)
        else:
            short_y = x
        y = nn.Conv(inputs=y,
                    features=planes,
                    kernel_size=(3, 3),
                    strides=(stride, stride),
                    padding=((0, 0), (0, 0)),
                    bias=False)
        y = nn.BatchNorm(y,
                         use_running_average=not train,
                         momentum=0.9,
                         epsilon=1e-5)
        y = jax.nn.relu(y)
        y = nn.Conv(inputs=y,
                    features=planes,
                    kernel_size=(3, 3),
                    strides=(stride, stride),
                    padding=((1, 1), (1, 1)),
                    bias=False)
        y = nn.BatchNorm(y,
                         use_running_average=not train,
                         momentum=0.9,
                         epsilon=1e-5)
        y = jax.nn.relu(y)
        y = nn.Conv(inputs=y,
                    features=expansion * planes,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding=((0, 0), (0, 0)),
                    bias=False)
        y += short_y
        return y


class PreActResNet(nn.Module):
    def apply(self,
              x,
              block,
              num_blocks,
              expansion,
              num_outputs=10,
              train=True):
        in_planes = 64
        y = nn.Conv(inputs=x,
                    features=in_planes,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=((1, 1), (1, 1)),
                    bias=False)
        y, in_planes = self._block(inputs=y,
                                   block=block,
                                   in_planes=in_planes,
                                   planes=64,
                                   num_blocks=num_blocks[0],
                                   stride=1,
                                   expansion=expansion,
                                   train=train)
        y, in_planes = self._block(inputs=y,
                                   block=block,
                                   in_planes=in_planes,
                                   planes=128,
                                   num_blocks=num_blocks[1],
                                   stride=2,
                                   expansion=expansion,
                                   train=train)
        y, in_planes = self._block(inputs=y,
                                   block=block,
                                   in_planes=in_planes,
                                   planes=256,
                                   num_blocks=num_blocks[2],
                                   stride=2,
                                   expansion=expansion,
                                   train=train)
        y, in_planes = self._block(inputs=y,
                                   block=block,
                                   in_planes=in_planes,
                                   planes=512,
                                   num_blocks=num_blocks[3],
                                   stride=2,
                                   expansion=expansion,
                                   train=train)
        y = jax.nn.relu(y)
        y = nn.BatchNorm(y,
                         use_running_average=not train,
                         momentum=0.9,
                         epsilon=1e-5)
        y = flax.nn.avg_pool(y, window_shape=(4, 4))
        y = y.reshape((y.shape[0], -1))
        y = nn.Dense(y, features=num_outputs)
        return y

    def _block(self,
               inputs,
               block,
               in_planes,
               planes,
               num_blocks,
               stride,
               expansion,
               train=True):
        strides = [stride] + [1] * (num_blocks - 1)
        y = inputs
        for stride in strides:
            y = block(y,
                      in_planes=in_planes,
                      planes=planes,
                      stride=stride,
                      train=train)
            in_planes = planes * expansion
        return y, in_planes


PreActResNet18 = PreActResNet.partial(block=PreActBlock,
                                      num_blocks=[2, 2, 2, 2],
                                      expansion=1)
PreActResNet34 = PreActResNet.partial(block=PreActBlock,
                                      num_blocks=[3, 4, 6, 3],
                                      expansion=1)
PreActResNet50 = PreActResNet.partial(block=PreActBottleneck,
                                      num_blocks=[3, 4, 6, 3],
                                      expansion=4)
PreActResNet101 = PreActResNet.partial(block=PreActBottleneck,
                                       num_blocks=[3, 4, 23, 3],
                                       expansion=4)
PreActResNet152 = PreActResNet.partial(block=PreActBottleneck,
                                       num_blocks=[3, 8, 36, 3],
                                       expansion=4)
