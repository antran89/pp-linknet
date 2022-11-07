"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from modules import resnet_abn
from inplace_abn import ABN

from functools import partial

nonlinearity = partial(F.relu,inplace=True)

class Dblock_more_dilate(nn.Module):
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class Dblock2(nn.Module):
    def __init__(self,channel):
        super(Dblock2, self).__init__()
        out_channel = int(channel/4)
        self.dilate1 = nn.Conv2d(channel, out_channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = torch.cat((x, dilate1_out, dilate2_out, dilate3_out, dilate4_out), 1)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    
class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_more_dilate, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        
        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        #Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DinkNet34_logits(DinkNet34):
    '''
    This function does not return sigmoid values as the last layer.
    It is supposed to be used in fp16 training, FocalLoss training, etc.
    '''
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34_logits, self).__init__()

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out

class DinkNet34_context(DinkNet34_logits):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34_context, self).__init__()

        filters = [64, 128, 256, 1024]
        self.dblock = Dblock2(512)
        self.decoder4 = DecoderBlock(filters[3], filters[2])

class DinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DinkNet34_abn(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34_abn, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = resnet_abn.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock_abn(filters[3], filters[2])
        self.decoder3 = DecoderBlock_abn(filters[2], filters[1])
        self.decoder2 = DecoderBlock_abn(filters[1], filters[0])
        self.decoder1 = DecoderBlock_abn(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # e4.register_hook(print)#for debug
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out
    
class DecoderBlock_abn(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock_abn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = ABN(in_channels // 4)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = ABN(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = ABN(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        return x
    
class DinkNet34_context_abn(DinkNet34_abn):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34_context_abn, self).__init__()

        filters = [64, 128, 256, 1024]
        self.dblock = Dblock2(512)
        self.decoder4 = DecoderBlock_abn(filters[3], filters[2])

class PSPModule(nn.Module):
    def __init__(self, in_features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_features, size, len(sizes)) for size in sizes])
        self.bottleneck = nn.Conv2d(in_features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, in_features, size, N):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, int(in_features/N), kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class DinkNet34_psp(DinkNet34_logits):
    def __init__(self, num_classes=1, num_channels=3, sizes=(1, 2, 3, 6)):
        super(DinkNet34_psp, self).__init__()
        
        in_features = 512
        out_features = 512
        self.dblock = PSPModule(in_features, out_features, sizes)

class DinkNet34_psp_abn(DinkNet34_abn):
    def __init__(self, num_classes=1, num_channels=3, sizes=(1, 2, 3, 6)):
        super(DinkNet34_psp_abn, self).__init__()
        
        in_features = 512
        out_features = 512
        self.psp = PSPModule(in_features, out_features, sizes)
        
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # e4.register_hook(print)#for debug
        # Center
        e4 = self.psp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out

class DinkNet34_psp64_hdc(DinkNet34):
    def __init__(self, num_classes=1, num_channels=3, sizes=(1, 2, 3, 6)):
        super(DinkNet34_psp64_hdc, self).__init__()
        
        in_features = 512
        out_features = 512
        self.psp = PSPModule(in_features, out_features, sizes)
        # implementing 64 large size of context module: 64x64
        self.encoder4[0].conv1.stride=(1,1)
        self.encoder4[0].downsample[0].stride=(1,1)
        self.decoder4.deconv2.stride=(1,1)
        self.decoder4.deconv2.output_padding=(0,0)

        #implementing hybrid dilated convolution: layer3: (1,2,5)*2 = 6 blocks, layer4: (5,9,17) = 3 blocks
        dr = 2
        self.encoder3[1].conv1.dilation=(dr,dr)
        self.encoder3[1].conv1.padding=(dr,dr)
        self.encoder3[1].conv2.dilation=(dr,dr)
        self.encoder3[1].conv2.padding=(dr,dr)
        dr = 5
        self.encoder3[2].conv1.dilation=(dr,dr)
        self.encoder3[2].conv1.padding=(dr,dr)
        self.encoder3[2].conv2.dilation=(dr,dr)
        self.encoder3[2].conv2.padding=(dr,dr)
        dr = 2
        self.encoder3[4].conv1.dilation=(dr,dr)
        self.encoder3[4].conv1.padding=(dr,dr)
        self.encoder3[4].conv2.dilation=(dr,dr)
        self.encoder3[4].conv2.padding=(dr,dr)
        dr = 5
        self.encoder3[5].conv1.dilation=(dr,dr)
        self.encoder3[5].conv1.padding=(dr,dr)
        self.encoder3[5].conv2.dilation=(dr,dr)
        self.encoder3[5].conv2.padding=(dr,dr)

        dr = 2
        self.encoder4[1].conv1.dilation=(dr,dr)
        self.encoder4[1].conv1.padding=(dr,dr)
        self.encoder4[1].conv2.dilation=(dr,dr)
        self.encoder4[1].conv2.padding=(dr,dr)
        dr = 5
        self.encoder4[2].conv1.dilation=(dr,dr)
        self.encoder4[2].conv1.padding=(dr,dr)
        self.encoder4[2].conv2.dilation=(dr,dr)
        self.encoder4[2].conv2.padding=(dr,dr)
        
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.psp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out

class DinkNet34_psp64(DinkNet34_logits):
    def __init__(self, num_classes=1, num_channels=3, sizes=(1, 2, 3, 6)):
        super(DinkNet34_psp64, self).__init__()
        
        in_features = 512
        out_features = 512
        self.dblock = PSPModule(in_features, out_features, sizes)
        # implementing 64 large size of context module: 64x64
        self.encoder4[0].conv1.stride=(1,1)
        self.encoder4[0].downsample[0].stride=(1,1)
        self.decoder4.deconv2.stride=(1,1)
        self.decoder4.deconv2.output_padding=(0,0)

class DinkNet34_psp128(DinkNet34_logits):
    def __init__(self, num_classes=1, num_channels=3, sizes=(1, 2, 3, 6)):
        super(DinkNet34_psp128, self).__init__()
        
        in_features = 512
        out_features = 512
        self.dblock = PSPModule(in_features, out_features, sizes)
        # implementing 64 large size of context module: 64x64
        self.encoder3[0].conv1.stride=(1,1)
        self.encoder3[0].downsample[0].stride=(1,1)
        self.encoder4[0].conv1.stride=(1,1)
        self.encoder4[0].downsample[0].stride=(1,1)
        self.decoder4.deconv2.stride=(1,1)
        self.decoder4.deconv2.output_padding=(0,0)
        self.decoder3.deconv2.stride=(1,1)
        self.decoder3.deconv2.output_padding=(0,0)
        
class DinkNet34_abn_psp128(DinkNet34_abn):
    def __init__(self, num_classes=1, num_channels=3, sizes=(1, 2, 3, 6)):
        super(DinkNet34_abn_psp128, self).__init__()
        
        in_features = 512
        out_features = 512
        self.dblock = PSPModule(in_features, out_features, sizes)
        # implementing 64 large size of context module: 64x64
        self.encoder3[0].conv1.stride=(1,1)
        self.encoder3[0].downsample[0].stride=(1,1)
        self.encoder4[0].conv1.stride=(1,1)
        self.encoder4[0].downsample[0].stride=(1,1)
        self.decoder4.deconv2.stride=(1,1)
        self.decoder4.deconv2.output_padding=(0,0)
        self.decoder3.deconv2.stride=(1,1)
        self.decoder3.deconv2.output_padding=(0,0)