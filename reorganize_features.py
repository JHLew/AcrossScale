from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg19
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.inception import inception_v3
import torch
import torch.nn as nn


# RF
# AlexNet 195

# VGG-11 150
# VGG-13 156
# VGG-16 212
# VGG-19 268

# resnet18 N/A (v2 435)
# resnet34 N/A (v2 899)
# resnet50 483 (v2 427)
# resnet101 1027 (v2 971)
# resnet152 1507
# inception-v3 1311


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, flatten_first=True):
        super(ProjectionHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_dim, 128)
        self.flatten = flatten_first

    def forward(self, x):
        if self.flatten:
            x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class AlexNet_feats(nn.Module):
    def __init__(self, feats, linear_feats):
        super(AlexNet_feats, self).__init__()
        self.features = feats
        self.linear_feats = linear_feats

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear_feats(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, alexnet, dropout=False):
        super(AlexNet, self).__init__()
        features = []
        features += [alexnet.features, alexnet.avgpool]
        linear_features = []
        if dropout:
            linear_features += [alexnet.classifier[:-1]]
        else:
            linear_features += [alexnet.classifier[1: 3], alexnet.classifier[4: -1]]
        features = nn.Sequential(*features)
        linear_features = nn.Sequential(*linear_features)
        self.features = AlexNet_feats(features, linear_features)
        self.fc = alexnet.classifier[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)

        return x


class VGG_feats(nn.Module):
    def __init__(self, feats, linear_feats):
        super(VGG_feats, self).__init__()
        self.features = feats
        self.linear_feats = linear_feats

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear_feats(x)

        return x


class VGG(nn.Module):
    def __init__(self, vgg, dropout=False):
        super(VGG, self).__init__()
        features = []
        features += [vgg.features, vgg.avgpool]
        features = nn.Sequential(*features)
        linear_features = []
        if dropout:
            linear_features += [vgg.classifier[:-1]]
        else:
            linear_features += [vgg.classifier[:2], vgg.classifier[3: 5]]
        linear_features = nn.Sequential(*linear_features)

        self.features = VGG_feats(features, linear_features)
        self.fc = vgg.classifier[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)

        return x


class ResNet(nn.Module):
    def __init__(self, resnet):
        super(ResNet, self).__init__()
        features = []
        features += [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool]
        self.features = nn.Sequential(*features)
        self.fc = resnet.fc

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


