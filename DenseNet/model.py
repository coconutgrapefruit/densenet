import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNet(nn.Module):
    def __init__(self, ini_channel, num_outs, config):
        super(DNet, self).__init__()
        blocks = [Initial(ini_channel)]
        num_channels = ini_channel

        for block, block_config in config:
            if block == 'Dense':
                blocks.append(DBlock(num_channels, block_config))
                for layer_config in block_config:
                    num_channels = num_channels + layer_config[1]
            elif block == 'Transition':
                blocks.append(Transition(num_channels, block_config))
                num_channels = block_config[0]
            else:
                raise Exception('unknown block type')

        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(num_channels, num_outs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.blocks(x)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Before entering dense blocks
class Initial(nn.Module):
    def __init__(self, out_channel):
        super(Initial, self).__init__()
        self.conv = nn.Conv2d(3, out_channel, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        return out

class Transition(nn.Module):
    def __init__(self, in_channel, config):
        super(Transition, self).__init__()
        out_channel, pool_size = config
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=pool_size)

    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        return out


class DBlock(nn.Module):
    def __init__(self, in_channel, config):
        super(DBlock, self).__init__()
        size = 0
        layers = []
        for i in range(len(config)):
            layers.append(Dlayer(in_channel + size, config[i]))
            size = size + config[i][1]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Dlayer(nn.Module):
    def __init__(self, in_channel, config):
        super(Dlayer, self).__init__()
        bn_k, growth_k, dropout = config
        self.dropout = dropout
        self.bn_bn = nn.BatchNorm2d(in_channel)
        self.bn_relu = nn.ReLU(inplace=True)
        self.bn_conv = nn.Conv2d(in_channel, bn_k, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(bn_k)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(bn_k, growth_k, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        bn_out = self.bn_conv(self.bn_relu(self.bn_bn(x)))
        out = self.conv(self.relu(self.bn(bn_out)))
        out = F.dropout(out, p=self.dropout, training=self.training)
        return torch.cat([x, out], 1)
