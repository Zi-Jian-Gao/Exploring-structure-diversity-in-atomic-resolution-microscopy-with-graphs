import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x
    
class Backbone(nn.Module):
    def __init__(self, in_channels=2):
        super(Backbone, self).__init__()
        # Downsample
        self.block1 = ConvBnRelu(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = ConvBnRelu(32, 32*2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = ConvBnRelu(32*2, 32*4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = ConvBnRelu(32*4, 32*16)
        
        # Upsample
        self.up5 = nn.Upsample(scale_factor=2)
        self.block5 = ConvBnRelu(32*(16+4), 32*4)
        self.up6 = nn.Upsample(scale_factor=2)
        self.block6 = ConvBnRelu(32*(4+2), 32*2)
        self.up7 = nn.Upsample(scale_factor=2)
        self.block7 = ConvBnRelu(32*(2+1), 32)
        
        # Out
        self.o1 = ConvBnRelu(32*16, 32)
        self.o2 = ConvBnRelu(32*4, 32)
        self.o3 = ConvBnRelu(32*2, 32)

    def forward(self, x):
        # Downsample
        bo1 = self.block1(x)
        po1 = self.pool1(bo1)
        bo2 = self.block2(po1)
        po2 = self.pool2(bo2)
        bo3 = self.block3(po2)
        po3 = self.pool3(bo3)
        bo4 = self.block4(po3)
        
        # Upsample
        uo5 = self.up5(bo4)
        bo5 = self.block5(torch.concat([uo5, bo3], dim=1))
        uo6 = self.up6(bo5)
        bo6 = self.block6(torch.concat([uo6, bo2], dim=1))
        uo7 = self.up7(bo6)
        bo7 = self.block7(torch.concat([uo7, bo1], dim=1))
        
        # Out
        o1 = self.o1(bo4)
        o2 = self.o2(bo5)
        o3 = self.o3(bo6) 
        
        return o1, o2, o3, bo7
    
class C_FCRN_Aux(nn.Module):
    def __init__(self, in_channels=3):
        super(C_FCRN_Aux, self).__init__()
        self.backbone = Backbone(in_channels)
        self.conv1 = nn.Conv2d(32, 1, 1, 1)
        self.conv2 = nn.Conv2d(32, 1, 1, 1)
        self.conv3 = nn.Conv2d(32, 1, 1, 1)
        self.conv4 = nn.Conv2d(32, 1, 1, 1)
        self.m = torch.nn.Sigmoid()
        
    def forward(self, x):
        o1, o2, o3, bo7 = self.backbone(x)
        den1 = self.m(self.conv1(o1))
        den2 = self.m(self.conv2(o2))
        den3 = self.m(self.conv3(o3))
        den4 = self.m(self.conv4(bo7))
        
        return den1, den2, den3, den4
