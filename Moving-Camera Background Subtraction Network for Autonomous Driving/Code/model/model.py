import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torchvision.models as models
import sys
sys.path.append(r'E:\Students\Jianli Wei\Background Subtraction\model')
from backbones import *

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


# Avaiable network backbones
def build_contextpath(name):
    model = {
        'GoogleNet': GoogleNet(pretrained=True),
        'Xception39': Xception39(pretrained=True),
        'MobileNet': MobileNet(pretrained=True),
        'Resnet18': Resnet18(pretrained=True),
        'Resnet34': Resnet18(pretrained=True),
        'Resnet50': Resnet50(pretrained=True),
        'Resnet101': Resnet101(pretrained=True)
    }
    return model[name]


# conv+bn+relu with stride=2
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


# deconv+bn+relu as decoder block      
class ConvTransposeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                padding=0, output_padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=output_padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        assert self.in_channels==input.size(1)
        x = self.deconv(input)
        return self.relu(self.bn(x))


class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


# provide 2x upsampling
class UP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                output_padding=0, bias=True):
        super().__init__()
        self.deconvblock1 = ConvTransposeBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                stride=stride, padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, input):
        x = self.deconvblock1(input)
        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) #1*1 conv
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))

        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # Xception39 3032 = 256(from spatial path) + 728(from context path) + 2048(from context path)
        # GoogleNet  2112 = 256(from spatial path) + 832(from context path) + 1024(from context path)
        # MobileNet  1632 = 256(from spatial path) + 96(from context path)  + 1280(from context path)
        # Resnet18   1024 = 256(from spatial path) + 256(from context path) + 512(from context path)
        # Resnet34   1024 = 256(from spatial path) + 256(from context path) + 512(from context path)
        # Resnet50   3328 = 256(from spatial path) + 1024(from context path) + 2048(from context path)
        # Resnet101  3328 = 256(from spatial path) + 1024(from context path) + 2048(from context path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)    #keeps dims unchanged
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):

        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path, Deconvolution=False):
        super().__init__()
        self.Deconv = Deconvolution
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        if context_path == 'GoogleNet':    # build attention refinement module for GoogleNet
            ARM1_channels,  ARM2_channels = 832, 1024
        elif context_path == 'Xception39':    # build attention refinement module for Xception39
            ARM1_channels,  ARM2_channels = 728, 2048
        elif context_path == 'MobileNet':    # build attention refinement module for MobileNet
            ARM1_channels,  ARM2_channels = 96, 1280
        elif context_path == 'Resnet18':    # build attention refinement module for Resnet18
            ARM1_channels,  ARM2_channels = 256, 512
        elif context_path == 'Resnet34':    # build attention refinement module for Resnet34
            ARM1_channels,  ARM2_channels = 256, 512
        elif context_path == 'Resnet50':    # build attention refinement module for Resnet50
            ARM1_channels,  ARM2_channels = 1024, 2048
        elif context_path == 'Resnet101':    # build attention refinement module for Resnet101
            ARM1_channels,  ARM2_channels = 1024, 2048
        else:
            print('Error: unspport context_path network \n')
        
        SP_channels = 256
        FFM_channels = SP_channels+ARM1_channels+ARM2_channels    
        
        # build ARM and FFM
        self.attention_refinement_module1 = AttentionRefinementModule(ARM1_channels, ARM1_channels)
        self.attention_refinement_module2 = AttentionRefinementModule(ARM2_channels, ARM2_channels)
        # supervision block
        self.supervision1 = nn.Conv2d(in_channels=ARM1_channels, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=ARM2_channels, out_channels=num_classes, kernel_size=1)
        # build feature fusion module
        self.feature_fusion_module = FeatureFusionModule(FFM_channels, num_classes)


        # build upsampling deconvolution
        if Deconvolution:
            self.up1 = UP(in_channels=ARM1_channels, out_channels=ARM1_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False)
            self.up2 = UP(in_channels=ARM2_channels, out_channels=ARM2_channels, kernel_size=2, stride=2, padding=1, output_padding=1, bias=False)
            self.up3 = UP(in_channels=ARM2_channels, out_channels=ARM2_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False)
            self.up4 = UP(in_channels=num_classes, out_channels=num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False)
        
        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1, bias=False)
        
        # initialize weights
        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)
        # for part in self.mul_lr:
            # print(part)

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        H, W = input.size()[2:]
        
        # output of spatial path
        sx = self.saptial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        
        if self.Deconv:   
            # Take deconvolution as upsampling technique
            # upsampling
            cx1 = self.up1(cx1)
            cx2 = self.up3(self.up2(cx2))
            cx = torch.cat((cx1, cx2), dim=1)

            # output of feature fusion module
            result = self.feature_fusion_module(sx, cx)

            # upsampling
            result = self.up4(self.up4(self.up4(result)))
        else:   
            # Take bilinear interpolation as upsampling technique
            # upsampling
            cx1 = F.interpolate(cx1, size=sx.size()[-2:], mode='bilinear', align_corners=True)
            cx2 = F.interpolate(cx2, size=sx.size()[-2:], mode='bilinear', align_corners=True)
            cx = torch.cat((cx1, cx2), dim=1)

            # output of feature fusion module
            result = self.feature_fusion_module(sx, cx)

            # upsampling
            result = F.interpolate(result, size=(H,W), mode='bilinear', align_corners=True)
        
        result = self.conv(result)
        
        return result


if __name__ == "__main__":
    device='cpu'
    net = BiSeNet(2, 'MobileNet', Deconvolution=False).to(device)
    net.eval()
    in_ten = torch.zeros(2,3,240,400)
    out = net(in_ten)
    print(out.shape)