import os
import cv2
import logging
import numpy as np

import torch
import torch.nn as nn
import utils
from   utils import CONFIG
import networks
from   utils import comput_sad_loss, compute_connectivity_error, \
    compute_gradient_loss, compute_mse_loss
from networks.saliency_sampler import Saliency_Sampler

class Tester(object):

    def __init__(self, test_dataloader):

        self.test_dataloader = test_dataloader
        self.logger = logging.getLogger("Logger")

        self.model_config = CONFIG.model
        self.test_config = CONFIG.test
        self.log_config = CONFIG.log
        self.data_config = CONFIG.data

        self.build_model()
        self.resume_step = None

        utils.print_network(self.G, CONFIG.version)

        if self.test_config.checkpoint:
            self.logger.info('Resume checkpoint: {}'.format(self.test_config.checkpoint))
            self.restore_model(self.test_config.checkpoint)

    def build_model(self):
        self.G = networks.get_generator(encoder=self.model_config.arch.encoder, encoder2=self.model_config.arch.encoder2, decoder=self.model_config.arch.decoder)
        self.modelv = Saliency_Sampler().cuda()
        if not self.test_config.cpu:
            self.G.cuda()

    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
        :param resume_checkpoint: File name of checkpoint
        :return:
        """
        pth_path = os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path)
        self.G.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    def test(self):
        self.G = self.G.eval()
        mse_loss = 0
        sad_loss = 0
        conn_loss = 0
        grad_loss = 0
        fg_mse_loss, bg_mse_loss = 0,0
        fg_sad_loss, bg_sad_loss = 0,0
        fg_grad_loss, bg_grad_loss = 0,0

        test_num = 0

        with torch.no_grad():
            for image_dict in self.test_dataloader:
                trimap_org, img_org = image_dict['trimap_org'], image_dict['img_org']
                
                tri = trimap_org / 255.
                tri[tri > 0] = 1
                tri_rev = trimap_org / 255.
                tri_rev[tri_rev > 0.7] = 2
                tri_rev[tri_rev < 1] = 1
                tri_rev[tri_rev == 2] = 0
                
                trimap_org = trimap_org.permute(0,3,1,2)
                tri = tri.permute(0,3,1,2)
                tri_rev = tri_rev.permute(0,3,1,2)
                img_org, trimap_org, tri, tri_rev = img_org.type(torch.FloatTensor).cuda(), trimap_org.type(torch.FloatTensor).cuda(), tri.type(torch.FloatTensor).cuda(), tri_rev.type(torch.FloatTensor).cuda()
                
                trans1, trans2, mask1, mask2 = self.modelv(torch.cat((img_org, trimap_org/255.), 1),tri,tri_rev)
                trans1 = trans1[:,:3,:,:]
                trans2 = trans2[:,:3,:,:]

                mask1_vis = mask1.clone()
                mask1_vis[mask1_vis > 0.5] = 0
                mask2_vis = mask2.clone()
                mask2_vis[mask2_vis > 0.3] = 1
                trans1 = trans1 * (1-mask1_vis)
                trans2 = trans2 * (1-mask2_vis)
            
                image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
                alpha_shape, name = image_dict['alpha_shape'], image_dict['image_name']
                fg = image_dict['fg']
                bg = image_dict['bg']

                a,b = image.shape[2:]
                trans1 = nn.Upsample(size=(a,b), mode='bilinear')(trans1)
                trans2 = nn.Upsample(size=(a,b), mode='bilinear')(trans2)
                mask1 = nn.Upsample(size=(a,b), mode='bilinear')(mask1)
                mask2 = nn.Upsample(size=(a,b), mode='bilinear')(mask2)
                
                if not self.test_config.cpu:
                    image = image.cuda()
                    alpha = alpha.cuda()
                    trimap = trimap.cuda()
                    fg = fg.cuda()
                    bg = bg.cuda()
                #alpha_pred, _ = self.G(image, trimap, trans2, mask2)
                alpha_pred, bg_pred, info_dict = self.G(image, trimap, trans1, mask1, trans2, mask2)
                #delta = image - (alpha_pred * fg_pred + (1-alpha_pred) * bg_pred)
                #delta_fg = alpha_pred / (alpha_pred*alpha_pred + (1-alpha_pred)*(1-alpha_pred)) * delta
                #delta_bg = (1-alpha_pred) / (alpha_pred*alpha_pred + (1-alpha_pred)*(1-alpha_pred)) * delta

                TRIMAP = trimap
                if self.model_config.trimap_channel == 3:
                    trimap = trimap.argmax(dim=1, keepdim=True)

                alpha_pred[trimap == 2] = 1
                alpha_pred[trimap == 0] = 0

                trimap[trimap==2] = 255
                trimap[trimap==1] = 128
                
                TRIMAP[TRIMAP==2] = 255
                TRIMAP[TRIMAP==1] = 128

                for cnt in range(image.shape[0]):

                    h, w = alpha_shape
                    test_alpha = alpha[cnt, 0, ...].data.cpu().numpy() * 255
                    TEST_ALPHA = (alpha[cnt, ...].repeat(3,1,1))#.data.cpu().numpy()
                    
                    test_pred = alpha_pred[cnt, 0, ...].data.cpu().numpy() * 255
                    TEST_PRED = (alpha_pred[cnt, ...].repeat(3,1,1)).data.cpu().numpy()
                    test_pred = test_pred.astype(np.uint8)
                    test_trimap = trimap[cnt, 0, ...].data.cpu().numpy()
                    TEST_TRIMAP = TRIMAP[cnt, ...].data.cpu().numpy()

                    test_pred = test_pred[:h, :w]
                    test_trimap = test_trimap[:h, :w]
                    TEST_PRED = TEST_PRED[:,:h, :w]
                    TEST_ALPHA = TEST_ALPHA[:, :h, :w]
                    TEST_TRIMAP = TEST_TRIMAP[:, :h, :w]

                    bg_pred = bg_pred[cnt, ...]#(bg_pred[cnt,...].cpu().numpy()*255).astype(np.uint8)
                    bg = bg[cnt, ...]#(bg[cnt,...].cpu().numpy()*255).astype(np.uint8)
                    #fg_pred = fg_pred[cnt, ...]#(fg_pred[cnt,...].cpu().numpy()*255).astype(np.uint8)
                    #fg = fg[cnt, ...]#(fg[cnt,...].cpu().numpy()*255).astype(np.uint8)
                    
                    #BG_PRED = bg_pred[:,:h,:w]
                    #BG = bg[:,:h,:w]
                    #FG_PRED = fg_pred[:,:h,:w]
                    #FG = fg[:,:h,:w]

                    #print(BG_PRED.shape)
                    #print(TEST_PRED.shape)
                    #print(BG.shape)
                    #print(TEST_ALPHA.shape)

                    if self.test_config.alpha_path is not None:
                        cv2.imwrite(os.path.join(self.test_config.alpha_path, os.path.splitext(name[cnt])[0] + ".png"),
                                    test_pred)

                    mse_loss += compute_mse_loss(test_pred, test_alpha, test_trimap)
                    print(name, comput_sad_loss(test_pred, test_alpha, test_trimap)[0])
                    #bg_mse_loss += (torch.sum((torch.abs((BG_PRED-bg)) ** 2)*(1-TEST_ALPHA)) / \
                    #        (torch.sum((1-TEST_ALPHA)) + 1e-8)).data.cpu().numpy()
                    #compute_mse_loss(BG_PRED*(1-TEST_PRED), BG*(1-TEST_ALPHA), TEST_TRIMAP)
                    #fg_mse_loss += (torch.sum((torch.abs((FG_PRED-fg)) ** 2)*(TEST_ALPHA)) / \
                    #        (torch.sum(TEST_ALPHA) + 1e-8)).data.cpu().numpy()
                    #compute_mse_loss(FG_PRED*TEST_PRED, FG*TEST_ALPHA, TEST_TRIMAP)
                    #print(name, (torch.sum(torch.abs((BG_PRED-bg))*(1-TEST_ALPHA)) / 1000).data.cpu().numpy())
                    sad_loss += comput_sad_loss(test_pred, test_alpha, test_trimap)[0]
                    #bg_sad_loss += (torch.sum(torch.abs((BG_PRED-bg))*(1-TEST_ALPHA)) / 1000).data.cpu().numpy()
                    #fg_sad_loss += (torch.sum(torch.abs((FG_PRED-fg))*TEST_ALPHA) / 1000).data.cpu().numpy()
                    #comput_sad_loss(FG_PRED*TEST_PRED, fg*TEST_ALPHA, TEST_TRIMAP)[0]
                    if not self.test_config.fast_eval:
                        conn_loss += compute_connectivity_error(test_pred, test_alpha, test_trimap, 0.1)
                        grad_loss += compute_gradient_loss(test_pred, test_alpha, test_trimap)

                    test_num += 1

        self.logger.info("TEST NUM: \t\t {}".format(test_num))
        self.logger.info("MSE: \t\t {}".format(mse_loss / test_num))
        self.logger.info("SAD: \t\t {}".format(sad_loss / test_num))
        #self.logger.info("BG_SAD: \t\t {}".format(bg_sad_loss / test_num))
        #self.logger.info("FG_SAD: \t\t {}".format(fg_sad_loss / test_num))
        #self.logger.info("BG_MSE: \t\t {}".format(bg_mse_loss / test_num))
        #self.logger.info("FG_MSE: \t\t {}".format(fg_mse_loss / test_num))
        if not self.test_config.fast_eval:
            self.logger.info("GRAD: \t\t {}".format(grad_loss / test_num))
            self.logger.info("CONN: \t\t {}".format(conn_loss / test_num))
