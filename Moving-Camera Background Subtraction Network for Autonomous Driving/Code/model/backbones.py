import torch
import torch.nn as nn

import torchvision.models as models
import sys
sys.path.append(r'E:\Students\Jianli Wei\Background Subtraction\model')
import xception as xception

'''
Avaible network backbones are

Xception39,
GoogleNet,
MobileNet,
Resnet18,
Resnet34,
Resnet50,
Resnet101,

'''


class Xception39(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = xception.xception(pretrained=pretrained)
        self.layer1 = nn.Sequential(self.features.conv1,
                                    self.features.bn1,
                                    self.features.relu,
                                    self.features.conv2,
                                    self.features.bn2,
                                    self.features.relu,
                                    self.features.block1
                                    )
        self.layer2 = nn.Sequential(self.features.block2)
        self.layer3 = nn.Sequential(self.features.block3,
                                    self.features.block4,
                                    self.features.block5,
                                    self.features.block6,
                                    self.features.block7,
                                    self.features.block8,
                                    self.features.block9,
                                    self.features.block10,
                                    self.features.block11
                                    )
        self.layer4 = nn.Sequential(self.features.block12,
                                    self.features.conv3,
                                    self.features.bn3,
                                    self.features.relu,
                                    self.features.conv4,
                                    self.features.bn4,
                                    self.features.relu
                                    )
        
    def forward(self, input):
        feature1 = self.layer1(input)   # 1 / 4
        feature2 = self.layer2(feature1)    # 1 / 8
        feature3 = self.layer3(feature2)    # 1 / 16
        feature4 = self.layer4(feature3)    # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class GoogleNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.googlenet(pretrained=pretrained)
        self.layer1 = nn.Sequential(self.features.conv1,
                                    self.features.maxpool1,
                                    self.features.conv2,
                                    self.features.conv3
                                    )
        self.layer2 = nn.Sequential(self.features.maxpool2,
                                    self.features.inception3a,
                                    self.features.inception3b
                                    )
        self.layer3 = nn.Sequential(self.features.maxpool3,
                                    self.features.inception4a,
                                    self.features.inception4b,
                                    self.features.inception4c,
                                    self.features.inception4d,
                                    self.features.inception4e,
                                    )
        self.layer4 = nn.Sequential(self.features.maxpool4,
                                    self.features.inception5a,
                                    self.features.inception5b
                                    )
    
    def forward(self, input):
        feature1 = self.layer1(input)   # 1 / 4
        feature2 = self.layer2(feature1)    # 1 / 8
        feature3 = self.layer3(feature2)    # 1 / 16
        feature4 = self.layer4(feature3)    # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class MobileNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.mobilenet_v2(pretrained=pretrained).features
        self.layer1 = nn.Sequential(self.features[0],
                                    self.features[1],
                                    self.features[2],
                                    self.features[3]
                                    )
        self.layer2 = nn.Sequential(self.features[4],
                                    self.features[5],
                                    self.features[6]
                                    )
        self.layer3 = nn.Sequential(self.features[7],
                                    self.features[8],
                                    self.features[9],
                                    self.features[10],
                                    self.features[11],
                                    self.features[12],
                                    self.features[13]
                                    )
        self.layer4 = nn.Sequential(self.features[14],
                                    self.features[15],
                                    self.features[16],
                                    self.features[17],
                                    self.features[18]
                                    )
    
    def forward(self, input):
        feature1 = self.layer1(input)   # 1 / 4
        feature2 = self.layer2(feature1)    # 1 / 8
        feature3 = self.layer3(feature2)    # 1 / 16
        feature4 = self.layer4(feature3)    # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail        


class Resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        #print('resnet18 features:', self.features)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class Resnet34(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet34(pretrained=pretrained)
        #print('resnet18 features:', self.features)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class Resnet50(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet50(pretrained=pretrained)
        #print('resnet50 features:', self.features)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class Resnet101(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail