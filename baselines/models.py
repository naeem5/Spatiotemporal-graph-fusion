import torch
import torch.nn as nn


# 3D ResNet
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes, in_channels=3, zero_init_residual=True):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 2, 2))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.feature_dim = 512 * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual=True):
        for mod in self.modules():
            if isinstance(mod, nn.Conv3d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, nn.BatchNorm3d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)
        if zero_init_residual:
            for mod in self.modules():
                if isinstance(mod, BasicBlock3D):
                    nn.init.constant_(mod.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        downsample = None
        if stride != (1, 1, 1) or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm3d(planes * block.expansion))
        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.fc(self.extract_features(x))


def set_bn_eps(model, eps=1e-4):
    # the network is trained in fp16 with eps = 1e-4 in every normalisation layer
    for mod in model.modules():
        if hasattr(mod, "eps"):
            mod.eps = eps
    return model
