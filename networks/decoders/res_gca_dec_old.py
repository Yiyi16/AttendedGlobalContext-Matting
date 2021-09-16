import torch
from   networks.ops import GuidedCxtAtten, SpectralNorm
from   networks.decoders.res_shortcut_dec import ResShortCut_D_Dec


class ResGuidedCxtAtten_Dec(ResShortCut_D_Dec):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False):
        super(ResGuidedCxtAtten_Dec, self).__init__(block, layers, norm_layer, large_kernel)
        self.gca = GuidedCxtAtten(128, 128)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        im = mid_fea['image_fea']
        x = self.layer1(x) + fea5 # N x 256 x 32 x 32
        x = self.layer2(x) + fea4 # N x 128 x 64 x 64
        x, offset = self.gca(im, x, mid_fea['unknown']) # contextual attention
        x = self.layer3(x) + fea3 # N x 64 x 128 x 128
        
        #xfg = self.layer4fg(x) + fea2
        xbg = self.layer4bg(x) + fea2
        x0 = self.layer4(x) + fea2 # N x 32 x 256 x 256
        
        #xfg = self.conv1fg(xfg)
        #xfg = self.bn1(xfg)
        #xfg = self.leaky_relu(xfg) + fea1
        #xfg = self.conv2fg(xfg)

        xbg = self.conv1bg(xbg)
        xbg = self.bn1(xbg)
        xbg = self.leaky_relu(xbg) + fea1
        xbg = self.conv2bg(xbg)

        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.leaky_relu(x0) + fea1
        x0 = self.conv2(x0)

        alpha = (self.tanh(x0) + 1.0) / 2.0
        #fg = (self.tanh(xfg)+1.0)/2.0
        bg = torch.sigmoid(xbg)#(self.tanh(xbg) + 1.0) / 2.0

        return alpha, bg, {'offset_1': mid_fea['offset_1'], 'offset_2': offset}

