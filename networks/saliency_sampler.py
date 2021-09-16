import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import numpy as np
import random
import cv2



def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


class Saliency_Sampler(nn.Module):
    #def __init__(self,task_input_size,saliency_input_size):
    def __init__(self):
        super(Saliency_Sampler, self).__init__()
        
        #self.hi_res = task_network
        self.grid_size = 21#31
        self.padding_size = 20#30
        self.global_size = self.grid_size+2*self.padding_size

        # self.input_size = saliency_input_size
        # self.input_size_net = task_input_size
        #self.conv_last = nn.Conv2d(256,1,kernel_size=1,padding=0,stride=1)
        gaussian_weights = torch.FloatTensor(makeGaussian(2*self.padding_size+1, fwhm = 13))
        # Spatial transformer localization-network
        #self.localization = saliency_network
        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.padding_size+1,2*self.padding_size+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        #self.P_basis = torch.zeros(2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)
        
        self.P_basis = torch.zeros(2,self.global_size, self.global_size).cuda()
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k,i,j] = k*(i-self.padding_size)/(self.grid_size-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_size-1.0)
                    #self.P_basis[k,i,j] = k*(i-self.padding_size)+(1.0-k)*(j-self.padding_size)

    def create_grid(self, x, x_ori):
        P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cuda(),requires_grad=False)

        P[0,:,:,:] = self.P_basis.cuda()
        P = P.expand(x.size(0),2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)
        

        x_cat = torch.cat((x,x),1)
        p_filter = self.filter(x)

        x_mul = torch.mul(P,x_cat).view(-1,1,self.global_size,self.global_size)

        all_filter = self.filter(x_mul).view(-1,2,self.grid_size,self.grid_size)


        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)

        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter

        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)
        ygrids = torch.clamp(ygrids,min=-1,max=1)


        xgrids = xgrids.view(-1,1,self.grid_size,self.grid_size)
        ygrids = ygrids.view(-1,1,self.grid_size,self.grid_size)

        grid = torch.cat((xgrids,ygrids),1)

        grid = nn.Upsample(size=(512,512), mode='bilinear')(grid)
        #grid = nn.Upsample(size=(x_ori.shape[2],x_ori.shape[3]), mode='bilinear')(grid)
        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        return grid

    def forward(self, x, tri, tri_rev):
        
        x_one = x[:,:,:,:]
        x_ori = x[:,:,:,:]
        
        mask = x[:,3,:,:]
        mask = mask.view(-1,1,x_ori.shape[2], x_ori.shape[3])
        
        x_one = nn.AdaptiveAvgPool2d((600,600))(x_one)
        x_ori = nn.AdaptiveAvgPool2d((600,600))(x_ori)
        mask = nn.AdaptiveAvgPool2d((600,600))(mask)
        
        x_ori = x_ori.view(-1, x_ori.shape[1], x_ori.shape[2], x_ori.shape[3])
        x_one = x_one.view(-1, x_one.shape[1], x_one.shape[2], x_one.shape[3])
        
      
        #xs = x_one[:,3,:,:]
        #xs[xs > 0] = 1
        #xs[xs < 1] = 0
        xs = tri
        xs = nn.AdaptiveAvgPool2d((600,600))(xs)
        xs = xs.view(-1, 1, x_ori.shape[2], x_ori.shape[3])
   
        xs = nn.Upsample(size=(self.grid_size,self.grid_size), mode='bilinear')(xs)
        xs = xs.view(-1,self.grid_size*self.grid_size)
        s = nn.Softmax()(xs)
        xs = xs.view(-1,1,self.grid_size,self.grid_size)
        #output1 = nn.Upsample(size=(320, 320), mode='bilinear')(xs)
        #xs = nn.ReplicationPad2d(self.padding_size)(xs)
        xs = nn.ZeroPad2d(self.padding_size)(xs)
        #print(xs.shape)
        grid = self.create_grid(xs, x)
        
    
        x_sampled = F.grid_sample(x_ori, grid)#x_ori
        mask_sample = F.grid_sample(mask, grid)
        
        '''
        img=np.zeros((320, 320,1),np.uint8)
        for i in range(320+32):
            for j in range(320+32):
                if i % 32 == 16 and j % 32 ==16:
                    x1, y1 = i,j
                    if i >= 32:
                        x2, y2 = i-32, j
                        cv2.line(img, (x1,y1), (x2,y2), (255,255,255),1)
                    if j >= 32:
                        x2, y2 = i, j-32
                        cv2.line(img, (x1,y1), (x2,y2), (255,255,255),1)

        img_t = torch.from_numpy(img.astype(np.float32)).cuda()
        img_t = img_t.view(-1,1,320,320)
        grid_vis = F.grid_sample(img_t, grid)
        cv2.imwrite('line1.png',grid_vis.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8))
        '''
        #mask_sample[mask_sample == 1.] = 0.
        #mask_sample[mask_sample > 0] = 1.
        
        # resize bg
        #xs1 = x_one[:,3,:,:]
        #xs1[xs1 == 1] = 2
        #xs1[xs1 < 1] = 1
        #xs1[xs1 == 2] = 0
        xs1 = tri_rev
        xs1 = nn.AdaptiveAvgPool2d((600,600))(xs1)
        xs1 = xs1.view(-1, 1, x_ori.shape[2], x_ori.shape[3])
   
        xs1 = nn.Upsample(size=(self.grid_size,self.grid_size), mode='bilinear')(xs1)
        xs1 = xs1.view(-1,self.grid_size*self.grid_size)
        xs1 = xs1.view(-1,1,self.grid_size,self.grid_size)
        #output2 = nn.Upsample(size=(320, 320), mode='bilinear')(xs1)
        xs1 = nn.ZeroPad2d(self.padding_size)(xs1)
        grid1 = self.create_grid(xs1, x)
       
        x_sampled1 = F.grid_sample(x_ori, grid1)#x_ori
        mask_sample1 = F.grid_sample(mask, grid1)
        
        '''
        img_t = torch.from_numpy(img.astype(np.float32)).cuda()
        img_t = img_t.view(-1,1,320,320)
        grid1_vis = F.grid_sample(img_t, grid1)
        cv2.imwrite('line2.png',grid1_vis.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8))
        '''

        '''
        if random.random()>p:
            s = random.randint(64, 320)
            x_sampled = nn.AdaptiveAvgPool2d((s,s))(x_sampled)
            x_sampled = nn.Upsample(size=(self.input_size_net,self.input_size_net),mode='bilinear')(x_sampled)
        '''
        
        return x_sampled,x_sampled1,mask_sample,mask_sample1
