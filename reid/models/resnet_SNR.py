"""
See https://github.com/microsoft/SNR
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet_SNR', 'resnet50_snr', 'resnet18_snr']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


#SNR
class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""
    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels //
                             reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates,
                             kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x) # pool5
        x = self.fc1(x) # W1
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x) # W2
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates: return x
        return input * x, input * (1 - x), x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Conv1x1nonLinear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1nonLinear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x


class ResNet_SNR_Causality(nn.Module):
    def __init__(self, block, layers, depth=50, num_classes=0,norm = True):
        self.inplanes = 64
        super(ResNet_SNR_Causality, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if depth==50:
            self.layer4[0].conv2.stride = (1, 1)
            self.layer4[0].downsample[0].stride = (1, 1)
            self.feat_bn = nn.BatchNorm1d(2048, affine=False)
            # IN bridge:
            self.IN1 = nn.InstanceNorm2d(256, affine=True)
            self.IN2 = nn.InstanceNorm2d(512, affine=True)
            self.IN3 = nn.InstanceNorm2d(1024, affine=True)
            self.IN4 = nn.InstanceNorm2d(2048, affine=True)
            # SE for selection:
            self.style_reid_laye1 = ChannelGate_sub(256, num_gates=256, return_gates=False,
                    gate_activation='sigmoid', reduction=16, layer_norm=False)
            self.style_reid_laye2 = ChannelGate_sub(512, num_gates=512, return_gates=False,
                    gate_activation='sigmoid', reduction=16, layer_norm=False)
            self.style_reid_laye3 = ChannelGate_sub(1024, num_gates=1024, return_gates=False,
                    gate_activation='sigmoid', reduction=16, layer_norm=False)
            self.style_reid_laye4 = ChannelGate_sub(2048, num_gates=2048, return_gates=False,
                    gate_activation='sigmoid', reduction=16, layer_norm=False)
        elif depth==18:
            self.feat_bn = nn.BatchNorm1d(512, affine=False)
            # IN bridge:
            self.IN1 = nn.InstanceNorm2d(64, affine=True)
            self.IN2 = nn.InstanceNorm2d(128, affine=True)
            self.IN3 = nn.InstanceNorm2d(256, affine=True)
            self.IN4 = nn.InstanceNorm2d(512, affine=True)
            # SE for selection:
            self.style_reid_laye1 = ChannelGate_sub(64, num_gates=64, return_gates=False,
                    gate_activation='sigmoid', reduction=16, layer_norm=False)
            self.style_reid_laye2 = ChannelGate_sub(128, num_gates=128, return_gates=False,
                    gate_activation='sigmoid', reduction=16, layer_norm=False)
            self.style_reid_laye3 = ChannelGate_sub(256, num_gates=256, return_gates=False,
                    gate_activation='sigmoid', reduction=16, layer_norm=False)
            self.style_reid_laye4 = ChannelGate_sub(512, num_gates=512, return_gates=False,
                    gate_activation='sigmoid', reduction=16, layer_norm=False)
            
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        
        if num_classes>0: #有
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            # FC layers for stage-1-2-3:
            if depth==50:
                self.fc1 = nn.Linear(256, num_classes)
                self.fc2 = nn.Linear(512, num_classes)
                self.fc3 = nn.Linear(1024, num_classes)
            elif depth==18:
                self.fc1 = nn.Linear(64, num_classes)
                self.fc2 = nn.Linear(128, num_classes)
                self.fc3 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

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
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x) # bsize*2048*8*4
        x = self.maxpool(x) 

        x_1 = self.layer1(x)  # torch.Size([64, 256, 64, 32])
        x_1_ori = x_1
        x_IN_1 = self.IN1(x_1)
        x_style_1 = x_1 - x_IN_1 # delta feature
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_laye1(x_style_1)
        x_1 = x_IN_1 + x_style_1_reid_useful
        x_1_useless = x_IN_1 + x_style_1_reid_useless

        x_2 = self.layer2(x_1)  # torch.Size([64, 512, 32, 16])
        x_2_ori = x_2
        x_IN_2 = self.IN2(x_2)
        x_style_2 = x_2 - x_IN_2
        x_style_2_reid_useful, x_style_2_reid_useless, selective_weight_useful_2 = self.style_reid_laye2(x_style_2)
        x_2 = x_IN_2 + x_style_2_reid_useful
        x_2_useless = x_IN_2 + x_style_2_reid_useless

        x_3 = self.layer3(x_2)  # torch.Size([64, 1024, 16, 8])
        x_3_ori = x_3
        x_IN_3 = self.IN3(x_3)
        x_style_3 = x_3 - x_IN_3
        x_style_3_reid_useful, x_style_3_reid_useless, selective_weight_useful_3 =\
            self.style_reid_laye3(x_style_3)
        x_3 = x_IN_3 + x_style_3_reid_useful
        x_3_useless = x_IN_3 + x_style_3_reid_useless

        x_4 = self.layer4(x_3)  # torch.Size([64, 2048, 16, 8])

        x_4 = self.global_avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)
        
        pool5 = self.feat_bn(x_4)
        if self.norm:
            pool5_norm = F.normalize(pool5)
        if self.training:
            if self.num_classes>0: #无
                fc1_in_score = self.fc1(self.global_avgpool(x_IN_1).view(x_IN_1.size(0), -1))
                fc1_score = self.fc1(self.global_avgpool(x_1).view(x_1.size(0), -1))
                fc1_useless_score = self.fc1(self.global_avgpool(x_1_useless).view(x_1_useless.size(0), -1))
                
                fc2_in_score = self.fc2(self.global_avgpool(x_IN_2).view(x_IN_2.size(0), -1))
                fc2_score = self.fc2(self.global_avgpool(x_2).view(x_2.size(0), -1))
                fc2_useless_score = self.fc2(self.global_avgpool(x_2_useless).view(x_2_useless.size(0), -1))
                
                fc3_in_score = self.fc3(self.global_avgpool(x_IN_3).view(x_IN_3.size(0), -1))
                fc3_score = self.fc3(self.global_avgpool(x_3).view(x_3.size(0), -1))
                fc3_useless_score = self.fc3(self.global_avgpool(x_3_useless).view(x_3_useless.size(0), -1))
                
                score = self.fc(pool5)
                
                return score, pool5, pool5_norm, fc1_in_score, fc1_score, fc1_useless_score, \
                   fc2_in_score, fc2_score, fc2_useless_score, \
                   fc3_in_score, fc3_score, fc3_useless_score
                   
            return pool5, pool5_norm, self.global_avgpool(x_IN_1).view(x_IN_1.size(0), -1), \
                   self.global_avgpool(x_1).view(x_1.size(0), -1), \
                   self.global_avgpool(x_1_useless).view(x_1_useless.size(0), -1), \
                   self.global_avgpool(x_IN_2).view(x_IN_2.size(0), -1), \
                   self.global_avgpool(x_2).view(x_2.size(0), -1), \
                   self.global_avgpool(x_2_useless).view(x_2_useless.size(0), -1), \
                   self.global_avgpool(x_IN_3).view(x_IN_3.size(0), -1), \
                   self.global_avgpool(x_3).view(x_3.size(0), -1), \
                   self.global_avgpool(x_3_useless).view(x_3_useless.size(0), -1)
        else:
            if self.num_classes > 0: 
                return self.fc(pool5)  # for pacs
            return F.normalize(pool5)


def resnet50_snr(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SNR_Causality(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weight_dict = model_zoo.load_url(model_urls['resnet50'])
        weight_dict = {key:value for (key, value) in weight_dict.items() if key.count('fc')==0}
        model.load_state_dict(weight_dict, strict=False)
    return model


def resnet18_snr(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model for pacs.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SNR_Causality(BasicBlock, [2, 2, 2, 2], depth=18, **kwargs)
    if pretrained:
        weight_dict = model_zoo.load_url(model_urls['resnet18'])
        weight_dict = {key:value for (key, value) in weight_dict.items() if key.count('fc')==0}
        model.load_state_dict(weight_dict, strict=False)
    return model
