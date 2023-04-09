"""
Function: Spatial Transform Network Layer
Author: Zhan Dong Xu
Date: 2021/11/11
"""

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from net.grid_sample import bilinear_sample_noloop

#---------------------------------------------------#
#  STN Layer
#---------------------------------------------------#
class SpatialTransformer(nn.Module, ABC):
    def __init__(self,
                 feat_width,
                 feat_height,
                 x_bound=None,
                 y_bound=None,
                 bev_feat_ratio=4,
                 org_img_width = 1600,
                 org_img_height= 900,
                 use_mask = False,
                 deploy=False,
                 mask_path=None):
        super(SpatialTransformer, self).__init__()

        self.use_mask = use_mask
        self.mask_path = mask_path
        self.deploy = deploy

        # BEV size calculation
        if y_bound is None:
            y_bound = [-32.0, 32.0, 0.25]
        if x_bound is None:
            x_bound = [-64.0, 64.0, 0.25]
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.bev_width = int((x_bound[1] - x_bound[0])/(x_bound[2]))
        self.bev_height = int((y_bound[1] - y_bound[0])/(y_bound[2]))
        self.bev_feat_ratio = bev_feat_ratio
        self.bev_feat_width = int(self.bev_width/self.bev_feat_ratio)
        self.bev_feat_height = int(self.bev_height/self.bev_feat_ratio)
        self.feat_width, self.feat_height = feat_width, feat_height
        self.org_img_width, self.org_img_height = org_img_width, org_img_height

        # Perspective Transformation Frustum
        self.frustum = self._make_regular_grids()

        # Mask For Projection
        if self.mask_path is None:
            self.mask_path = "data/mask_nuscene.npy"
        self.masks = torch.from_numpy(np.load(self.mask_path))

    def _make_regular_grids(self):
        # making a single regular grid
        xs = torch.linspace(0, self.bev_feat_width-1, self.bev_feat_width, dtype=torch.float).view(1, self.bev_feat_width).expand(self.bev_feat_height, self.bev_feat_width) * self.bev_feat_ratio
        ys = torch.linspace(0, self.bev_feat_height-1, self.bev_feat_height, dtype=torch.float).view(self.bev_feat_height, 1).expand(self.bev_feat_height, self.bev_feat_width) * self.bev_feat_ratio
        ds = torch.tensor(1.0).view(1, 1).expand(self.bev_feat_height, self.bev_feat_width)
        frustum = torch.stack((xs, ys, ds), 0).view(3,-1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, feat, pm_matrix, debug= False):

        if debug:
            for i in range(feat.shape[0]):
                vis_feat = feat[i,0].detach().cpu().numpy()
                plt.imshow(vis_feat)
                plt.show()

        pm_pts = torch.bmm(pm_matrix, self.frustum.unsqueeze(0).repeat(feat.shape[0], 1, 1).to(pm_matrix.device))
        pm_grid = pm_pts[:,0:2, :] / (pm_pts[:,2, :].view(feat.shape[0], 1, -1) + 1e-8)
        pm_grid[:,0,:] = ( (pm_grid[:,0,:]/ (self.org_img_width - 1)) - 0.5 ) * 2
        pm_grid[:,1,:] = ( (pm_grid[:,1,:]/ (self.org_img_height - 1)) - 0.5 ) * 2
        sample_grid = pm_grid.view(feat.shape[0], 2, self.bev_feat_height, self.bev_feat_width).permute(0,2,3,1).contiguous()

        if not self.deploy:
            bev_feat = F.grid_sample(feat, sample_grid, align_corners=False,padding_mode="zeros")
        else:
            bev_feat = bilinear_sample_noloop(feat, sample_grid)

        if self.use_mask:
            batch_size = int( int(bev_feat.shape[0]) / int(self.masks.shape[0]) )
            mask_all_out = self.masks.to(bev_feat.device).unsqueeze(1).repeat(1,bev_feat.shape[1],1,1).repeat(batch_size,1,1,1)
            bev_feat = bev_feat * mask_all_out

        if debug:
            for i in range(feat.shape[0]):
                vis_bev_feat = bev_feat[i, 0].detach().cpu().numpy()
                plt.imshow(vis_bev_feat)
                plt.show()


        return bev_feat

if __name__ == '__main__':
    import yaml, time
    from net.resnet import resnet18, resnet34
    CFG_PATH = "../cfgs/ultrabev_stn_seq3.yml"
    MASK_PATH = "../data/mask_nuscene.npy"
    batch_size = 1

    #---------------------------------------------------#
    #  Parameter Setting
    #---------------------------------------------------#
    cfgs = yaml.safe_load(open(CFG_PATH))
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    cam_num = cfgs["dataloader"]["cam_num"]
    x_bound = cfgs["dataloader"]["x_bound"]
    y_bound = cfgs["dataloader"]["y_bound"]
    layer_id = cfgs["backbone"]["layer_id"]
    feat_ratio = cfgs["backbone"]["bev_feat_ratio"]
    use_mask = cfgs["backbone"]["use_mask"]

    #---------------------------------------------------#
    #  Input Defination
    #---------------------------------------------------#
    x = torch.randn(batch_size, cam_num, 3, net_input_height, net_input_width).cuda()
    pm_matrix = torch.randn(batch_size,cam_num, 3, 3).cuda()
    x = x.view(batch_size*cam_num,3,net_input_height, net_input_width)
    pm_matrix = pm_matrix.view(batch_size*cam_num, 3, 3)

    #---------------------------------------------------#
    #  Model Definations
    #---------------------------------------------------#
    res_type = cfgs["backbone"]["resnet_type"]
    layers_dim = cfgs["backbone"]["dim_array"]

    if res_type == 18:
        backbone = resnet18(pretrained=False, layers_dim=layers_dim).cuda()
    else:
        backbone = resnet34(pretrained=False,layers_dim=layers_dim).cuda()

    spatial_transformer = SpatialTransformer(feat_width= int(net_input_width/pow(2,layer_id + 1)),
                                             feat_height=int(net_input_height/pow(2,layer_id + 1)),
                                             x_bound = x_bound,
                                             y_bound = y_bound,
                                             bev_feat_ratio=feat_ratio,
                                             use_mask=use_mask,
                                             mask_path= MASK_PATH).cuda()

    feat = backbone(x, True)[layer_id]
    bev_feat = spatial_transformer(feat, pm_matrix)
    bev_feat = bev_feat.view(batch_size,cam_num, bev_feat.shape[-3],bev_feat.shape[-2],bev_feat.shape[-1])
    print(bev_feat.shape)

    #---------------------------------------------------#
    #  Inference Test
    #---------------------------------------------------#
    # inference speed
    for _ in range(10):
        tic = time.time()
        torch.cuda.synchronize()
        feat = backbone(x, True)[0]
        ipm_feat = spatial_transformer(feat, pm_matrix)
        torch.cuda.synchronize()
        print("inference time is %f" %(1000*(time.time() - tic)))


    