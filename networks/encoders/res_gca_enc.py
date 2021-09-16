import torch
import torch.nn as nn
import torch.nn.functional as F

from   utils import CONFIG
from   networks.encoders.resnet_enc import ResNet_D
from   networks.ops import GuidedCxtAtten, SpectralNorm


class ResGuidedCxtAtten(ResNet_D):

    def __init__(self, block, layers, norm_layer=None, late_downsample=False):
        super(ResGuidedCxtAtten, self).__init__(block, layers, norm_layer, late_downsample=late_downsample)
        first_inplane = 3 + CONFIG.model.trimap_channel
        self.shortcut_inplane = [first_inplane, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))

        self.guidance_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(16),
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(32),
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(32, 128, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(128)
        )
        self.guidance_head_trans = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(16),
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(32),
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(32, 128, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(128)
        )
        self.guidance_head_transback = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(16),
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(32),
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(32, 128, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(128)
        )
        
        self.gca = GuidedCxtAtten(128, 128)
        self.gca_trans = GuidedCxtAtten(128, 128)
        self.gca_transback = GuidedCxtAtten(128, 128)

        # finetune
        
        
        # initialize guidance head
        for layers in range(len(self.guidance_head)):
            m = self.guidance_head[layers]
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for layers in range(len(self.guidance_head_trans)):
            m = self.guidance_head_trans[layers]
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for layers in range(len(self.guidance_head_transback)):
            m = self.guidance_head_transback[layers]
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes)
        )

    def forward(self, x, xtrans, xtransback):
        out = self.conv1(x)    
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out) # N x 32 x 256 x 256
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out)
        
        out_trans = self.conv1_trans(xtrans)
        out_trans = self.bn1_trans(out_trans)
        out_trans = self.activation(out_trans)
        out_trans = self.conv2_trans(out_trans)
        out_trans = self.bn2_trans(out_trans)
        x1_trans = self.activation(out_trans) # N x 32 x 256 x 256
        out_trans = self.conv3_trans(x1_trans)
        out_trans = self.bn3_trans(out_trans)
        out_trans = self.activation(out_trans)
        
        out_transback = self.conv1_transback(xtransback)
        out_transback = self.bn1_transback(out_transback)
        out_transback = self.activation(out_transback)
        out_transback = self.conv2_transback(out_transback)
        out_transback = self.bn2_transback(out_transback)
        x1_transback = self.activation(out_transback) # N x 32 x 256 x 256
        out_transback = self.conv3_transback(x1_transback)
        out_transback = self.bn3_transback(out_transback)
        out_transback = self.activation(out_transback)
        
        im_fea = self.guidance_head(x[:,:3,...]) # downsample origin image and extract features
        im_fea_trans = self.guidance_head_trans(xtrans[:,:3,...])
        im_fea_transback = self.guidance_head_transback(xtransback[:,:3,...])
         
        unknown = F.interpolate(x[:,4:5,...], scale_factor=1/8, mode='nearest')
        unknown_trans = F.interpolate(xtrans[:,3:,...].eq(1.).float(), scale_factor=1/8, mode='nearest')
        unknown_transback = F.interpolate(xtransback[:,3:,...].eq(1.).float(), scale_factor=1/8, mode='nearest')

        x2 = self.layer1(out) # N x 64 x 128 x 128
        x3 = self.layer2(x2) # N x 128 x 64 x 64
        x2_trans = self.layer1_trans(out_trans)
        x3_trans = self.layer2_trans(x2_trans)
        x2_transback = self.layer1_transback(out_transback)
        x3_transback = self.layer2_transback(x2_transback)
        x3, offset = self.gca(im_fea, x3, unknown) # contextual attention
        x3_trans, offset_trans = self.gca_trans(im_fea_trans, x3_trans, unknown_trans) # contextual attention
        x3_transback, offset_transback = self.gca_transback(im_fea_transback, x3_transback, unknown_transback) # contextual attention
        x3 = torch.add(x3, 0.5*x3_trans)
        x3 = torch.add(x3, 0.5*x3_transback)
        x4 = self.layer3(x3) # N x 256 x 32 x 32
        out = self.layer_bottleneck(x4) # N x 512 x 16 x 16
        
        fea1 = self.shortcut[0](x) # input image and trimap
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)
  
        return out, {'shortcut':(fea1, fea2, fea3, fea4, fea5),
                       'image_fea':im_fea,
                       'unknown':unknown,
                       'offset_1':offset,
                       'offset_f':offset_trans,
                       'offset_b':offset_transback}



if __name__ == "__main__":
    from networks.encoders.resnet_enc import BasicBlock
    m = ResGuidedCxtAtten(BasicBlock, [3, 4, 4, 2])
    for m in m.modules():
        print(m)
