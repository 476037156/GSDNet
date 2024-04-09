import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# from attention import CBAM


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(),
                                 nn.Linear(gate_channels, gate_channels // reduction_ratio),
                                 nn.ReLU(),
                                 nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.max_weight = nn.Sequential(Flatten(),
                                        nn.Linear(gate_channels, gate_channels // reduction_ratio),
                                        # nn.BatchNorm1d(gate_channels // reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(gate_channels // reduction_ratio, 1)
                                        )
        self.mean_weight = nn.Sequential(Flatten(),
                                         nn.Linear(gate_channels, gate_channels // reduction_ratio),
                                         # nn.BatchNorm1d(gate_channels // reduction_ratio),
                                         nn.ReLU(),
                                         nn.Linear(gate_channels // reduction_ratio, 1)
                                         )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
                channel_att_raw = channel_att_raw * self.mean_weight(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
                channel_att_raw = channel_att_raw * self.max_weight(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ACBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ACBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x):
        x_out = self.ChannelGate(x)

        return x_out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=8631, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.rep_dim = 512 * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.cm1 = nn.Sequential(
            ACBAM(64 * block.expansion),
            nn.Conv2d(64 * block.expansion, 512 * block.expansion, 1),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.fc_1 = nn.Linear(512 * block.expansion, 128)
        self.fc_1_1 = nn.Linear(512 * block.expansion, num_classes)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.cm2 = nn.Sequential(
            ACBAM(128 * block.expansion),
            nn.Conv2d(128 * block.expansion, 512 * block.expansion, 1),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.fc_2 = nn.Linear(512 * block.expansion, 128)
        self.fc_2_2 = nn.Linear(512 * block.expansion, num_classes)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.cm3 = nn.Sequential(
            ACBAM(256 * block.expansion),
            nn.Conv2d(256 * block.expansion, 512 * block.expansion, 1),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.fc_3 = nn.Linear(512 * block.expansion, 128)
        self.fc_3_3 = nn.Linear(512 * block.expansion, num_classes)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.cm4 = nn.Sequential(
            ACBAM(512 * block.expansion),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_4 = nn.Linear(512 * block.expansion, 128)
        self.fc_4_4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        out_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        a = self.cm1(x)
        a_feature = self.fc_1(a)
        feature_list.append(a_feature)
        a_out = self.fc_1_1(a)
        out_list.append(a_out)
        pred_teacher = a_out
        feat_teacher = a_feature

        x = self.layer2(x)
        b = self.cm2(x)
        b_feature = self.fc_2(b)
        feature_list.append(b_feature)
        b_out = self.fc_2_2(b)
        out_list.append(b_out)
        pred_teacher += b_out
        feat_teacher += b_feature

        x = self.layer3(x)
        c = self.cm3(x)
        c_feature = self.fc_3(c)
        feature_list.append(c_feature)
        c_out = self.fc_3_3(c)
        out_list.append(c_out)
        pred_teacher += c_out
        feat_teacher += c_feature

        x = self.layer4(x)
        d = self.cm4(x)
        d_feature = self.fc_4(d)
        feature_list.append(d_feature)
        d_out = self.fc_4_4(d)
        out_list.append(d_out)
        pred_teacher += d_out
        feat_teacher += d_feature

        return out_list, feature_list, pred_teacher, feat_teacher
