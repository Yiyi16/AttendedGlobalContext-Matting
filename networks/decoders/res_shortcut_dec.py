from   networks.decoders.resnet_dec import ResNet_D_Dec


class ResShortCut_D_Dec(ResNet_D_Dec):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResShortCut_D_Dec, self).__init__(block, layers, norm_layer, large_kernel,
                                                late_downsample=late_downsample)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x = self.layer3(x) + fea3
        xfg = self.layer4fg(x) + fea2
        xbg = self.layer4bg(x) + fea2
        x0 = self.layer4(x) + fea2

        xfg = self.conv1fg(xfg)
        xfg = self.bn1(xfg)
        xfg = self.leaky_relu(xfg) + fea1
        xfg = self.conv2fg(xfg)

        xbg = self.conv1bg(xbg)
        xbg = self.bn1(xbg)
        xbg = self.leaky_relu(xbg) + fea1
        xbg = self.conv2bg(xbg)
        
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.leaky_relu(x0) + fea1
        x0 = self.conv2(x0)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha, None

