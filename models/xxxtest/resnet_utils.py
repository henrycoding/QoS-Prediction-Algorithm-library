from abc import get_cache_token
from collections import OrderedDict

from torch import nn


class Linear(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 activation,
                 dropout=0.3,
                 personal_layer_name=None):
        super().__init__()
        if personal_layer_name is None:
            self.fc_layer = nn.Sequential(nn.Linear(in_size, out_size),
                                          activation(), nn.Dropout(dropout))
        else:
            self.fc_layer = nn.Sequential(
                OrderedDict({
                    personal_layer_name: nn.Linear(in_size, out_size),
                    "activation": activation()
                }))

    def forward(self, x):
        x = self.fc_layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size, addition=None):
        super().__init__()
        self.in_size, self.out_size = in_size, out_size
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_size != self.out_size


# 用来处理short cut
class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)
        self.shortcut = nn.Sequential(
            OrderedDict({
                'dense': nn.Linear(self.in_size, self.out_size),
                'dropout': nn.Dropout(0.3)
                # 'bn': nn.BatchNorm1d(self.out_size)
            })) if self.should_apply_shortcut else None

    @property
    def should_apply_shortcut(self):
        return self.in_size != self.out_size


# 来定义一个block


class ResNetBasicBlock(ResNetResidualBlock):
    def __init__(self, in_size, out_size, activation=nn.ReLU):
        super().__init__(in_size, out_size)
        self.blocks = nn.Sequential(
            nn.Linear(self.in_size, self.out_size),
            activation(),
            nn.Dropout(0.5),
            nn.Linear(self.out_size, self.out_size),
        )

class ResNetBasicBlock_V2(ResNetResidualBlock):
    def __init__(self, in_size, out_size, activation=nn.ReLU):
        super().__init__(in_size, out_size)
        self.blocks = nn.Sequential(
            nn.Linear(self.in_size, self.in_size),
            activation(),
            nn.Dropout(0.5),
            nn.Linear(self.in_size, self.out_size),
        )

# 定义一个resnet layer层，里面会有多个block
class ResNetLayer(nn.Module):
    """对于[128,64] n=2时 返回 128x64, 64x64
    """
    def __init__(self,
                 in_size,
                 out_size,
                 block=ResNetBasicBlock,
                 n=1,
                 activation=nn.ReLU):
        super().__init__()
        self.blocks = nn.Sequential(
            block(in_size, out_size, activation),
            *[block(out_size, out_size, activation) for _ in range(n - 1)])

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetLayer_v2(nn.Module):
    """对于[128,64] n=2时 返回 128x128, 128x64
    """
    def __init__(self,
                 in_size,
                 out_size,
                 block=ResNetBasicBlock_V2,
                 n=1,
                 activation=nn.ReLU):
        super().__init__()
        self.blocks = nn.Sequential(
            *[block(in_size, in_size, activation) for _ in range(n - 1)],
            block(in_size, out_size, activation))

    def forward(self, x):
        x = self.blocks(x)
        return x


# 由多个resnet layer组成encoder
class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by decreasing different layers with increasing features.
    """
    def __init__(self,
                 in_size=128,
                 blocks_sizes=[64, 32, 16],
                 deepths=[2, 2, 2],
                 activation=nn.ReLU,
                 block=ResNetBasicBlock):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Linear(in_size, self.blocks_sizes[0]),
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        print("LEFT:",[(in_size,out_size,n) for (in_size,out_size), n in zip(self.in_out_block_sizes, deepths)])
        self.blocks = nn.ModuleList([
            *[
                ResNetLayer(
                    in_size, out_size, n=n, activation=activation, block=block)
                for (in_size,
                     out_size), n in zip(self.in_out_block_sizes, deepths)
            ]
        ])

    # def forward(self, x):
    #     x = self.gate(x)
    #     accessories = []
    #     # layers
    #     for block in self.blocks:
    #         x = block(x)
    #         accessories.append(x)
    #     return x, accessories

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x,None


class ResNetEncoder_v2(nn.Module):
    """
    单纯适配ResNetLayer_v2
    """
    def __init__(self,
                 output_size=128,
                 blocks_sizes=[32, 64, 128],
                 deepths=[2, 2],
                 activation=nn.ReLU,
                 block=ResNetBasicBlock_V2):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        print("Right:",[(in_size,out_size,n) for (in_size,out_size), n in zip(self.in_out_block_sizes, deepths)])

        self.blocks = nn.ModuleList([
            *[
                ResNetLayer_v2(
                    in_size, out_size, n=n, activation=activation, block=block)
                for (in_size,
                     out_size), n in zip(self.in_out_block_sizes, deepths)
            ]
        ])

        # self.gate = nn.Sequential(
        #     nn.Linear(self.blocks_sizes[-1], output_size), activation(),
        #     nn.Dropout(0.5)
        # )

    def forward(self, x, accessories: list):
        for idx, block in enumerate(self.blocks):
            # if idx != 0:
            #     assert x.shape == accessories[-idx - 1].shape
            #     x += accessories[-idx - 1]
            x = block(x)
        # x = self.gate(x)
        return x


if __name__ == "__main__":
    m = ResNetEncoder()
    n = ResNetEncoder_v2()

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters()
                            if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    # print(m)
    # print(get_parameter_number(m))

    print(n)
    print(get_parameter_number(n))
