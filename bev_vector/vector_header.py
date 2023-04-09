"""
Function: Vector Header
Author: Zhan Dong Xu
Date: 2021/11/11
"""

from abc import ABC
import warnings

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False
USE_DIRECTION_INFO = True
DRAW_LINE = True

def resize(input_,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input_.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input_ size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input_, size, scale_factor, mode, align_corners)

class Conv2D_BatchNorm_Relu(nn.Module, ABC):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, dilation=1):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size,
                                                    padding=padding, stride=stride, bias=bias, dilation=dilation),
                                          nn.BatchNorm2d(n_filters),
                                          # nn.ReLU(inplace=True),
                                          nn.PReLU(),
                                          )
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Conv3x3(nn.Module, ABC):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module, ABC):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Output(nn.Module, ABC):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size//2, 3, 1, 1, dilation=1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size//2, in_size//4, 3, 1, 1, dilation=1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size//4, out_size, 1, 0, 1, acti = False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class VectorHeader(nn.Module, ABC):
    def __init__(self,
                 feat_channel,
                 cluster_feat_dim,
                 exist_condidence_loss = 0.1,
                 nonexist_confidence_loss = 1.0,
                 sisc_loss = 0.5,
                 disc_loss = 0.5,
                 grid_x=64,
                 grid_y=32,
                 x_size=512,
                 y_size=256,
                 delta_v=0.5,
                 delta_d=0.5,
                 thresh=0.3,
                 threshold_instance=0.01,
                 resize_ratio=8,
                 class_num=3,
                 threshold_non_exist=0.01,
                 resolution=0.125,
                 sample_distance=None,
                 threshold_remove=None,
                 angle_list=None,
                 up_sample_time = 1,
                 cluster_min_num = 8,
                 min_pt_num=None,
                 use_half_direct=True,
                 use_resize_conv_os=True,
                 ):
        super(VectorHeader, self).__init__()

        if min_pt_num is None:
            min_pt_num = [4, 4, 4]
        if threshold_remove is None:
            threshold_remove = [2.5, 2.5, 7.0]
        if sample_distance is None:
            sample_distance = [0.5, 0.5, 1.0]
        if angle_list is None:
            angle_list = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        self.angle_list = angle_list
        self.up_sample_time = up_sample_time
        self.cluster_min_num = cluster_min_num
        self.cls_num = class_num
        self.use_half_direct = use_half_direct

        # Angel List
        self.cls_index_list = list(range(len(self.angle_list)) )
        self.half_circle_num = int(len(self.cls_index_list)/2)
        self.direction_num = len(self.angle_list)

        # Vectorization Parameters
        self.resolution = resolution
        self.sample_distance = sample_distance
        self.thershold_remove = threshold_remove

        # Output Header Definations
        self.out_confidence = Output(feat_channel, 1)
        self.out_offset = Output(feat_channel, 2)
        self.out_instance = Output(feat_channel, cluster_feat_dim)
        self.out_cls = Output(feat_channel, class_num)

        # convolution defination
        self.use_resize_conv_os = use_resize_conv_os
        self.resize_convs = list()
        if self.use_resize_conv_os:
            for i in range(self.up_sample_time):
                self.resize_convs.append(ConvBlock(feat_channel, feat_channel))
            self.resize_convs = nn.ModuleList(self.resize_convs)

        if self.use_half_direct:
            self.out_direct = Output(feat_channel, self.half_circle_num)

        else:
            self.out_direct = Output(feat_channel, self.direction_num)

        # Size Definations
        self.grid_x = int(grid_x * np.exp2(self.up_sample_time) )
        self.grid_y = int(grid_y * np.exp2(self.up_sample_time) )

        self.x_size = x_size
        self.y_size = y_size

        # Cluster Loss Parameters
        self.delta_v = delta_v
        self.delta_d = delta_d

        # Postprocess Parameters
        self.thresh = thresh
        self.threshold_instance = threshold_instance
        self.feature_size = cluster_feat_dim
        self.resize_ratio = int(resize_ratio /  np.exp2(self.up_sample_time) )
        self.threshold_non_exist = threshold_non_exist

        self.grid_location = np.zeros((self.grid_y, self.grid_x, 2))
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                self.grid_location[y][x][0] = x
                self.grid_location[y][x][1] = y

        # Fixed Color
        self.color = [(0, 0, 255), (255, 0, 0), (0, 255, 0),
                      (0, 0, 255), (255, 255, 0), (255, 0, 255),
                      (0, 255, 255), (255, 255, 255), (100, 255, 0),
                      (100, 0, 255), (255, 100, 0), (0, 100, 255),
                      (255, 0, 100), (0, 255, 100), (0, 255, 127)]

        # loss
        self.exist_condidence_loss = exist_condidence_loss
        self.nonexist_confidence_loss =nonexist_confidence_loss
        self.sisc_loss = sisc_loss
        self.disc_loss = disc_loss

        # postprocess
        self.min_pt_num = min_pt_num

    def forward(self, feats):
        for i in range(self.up_sample_time):
            feats = resize(
                input_=feats,
                size=(feats.shape[2] * 2, feats.shape[3] * 2),
                mode='bilinear',
                align_corners=False
            )

            if self.use_resize_conv_os:
                feats = self.resize_convs[i](feats)

        out_confidence = self.out_confidence(feats)
        out_offset = self.out_offset(feats)
        out_instance = self.out_instance(feats)
        out_direct = self.out_direct(feats)
        out_cls= self.out_cls(feats)
        return [out_confidence, out_offset, out_instance,out_direct, out_cls]


    @staticmethod
    def cal_masked_cls_loss(cls_pred, cls_gt , positive_mask, focal_gamma= 1.0, focal_alpha = 1.0, use_focal=False ):
        cls_gt = cls_gt.permute((0, 2, 3, 1)).contiguous()
        cls_gt = cls_gt.view(-1, cls_gt.shape[-1])

        cls_pred = cls_pred.permute((0, 2, 3, 1)).contiguous()
        cls_pred = cls_pred.view(-1, cls_pred.shape[-1])

        # compute the actual focal loss
        cls_pred_softmax = F.softmax(cls_pred, dim=-1) + 1e-8
        if use_focal:
            weight = torch.pow(-cls_pred_softmax + 1., focal_gamma)
            focal = -focal_alpha * weight * torch.log(cls_pred_softmax)
        else:
            focal = -torch.log(cls_pred_softmax)
        loss_tmp = torch.sum(cls_gt * focal, dim=-1)
        loss_cls = torch.mean(loss_tmp[positive_mask])  # positive mask

        return loss_cls

    def cal_loss_vector(self,confidance_gt, offset_gt, instance_mask_gt, direct_gt, cls_gt, result):
        # update lane_detection_network
        exist_condidence_loss = 0
        nonexist_confidence_loss = 0
        cls_loss = 0
        direct_loss = 0
        offset_loss = 0
        sisc_loss = 0
        disc_loss = 0

        # hard sampling
        real_batch_size = result[0].shape[0]

        for (confidance, offset, feature, direct_pred, cls_pred) in [result]:
            # calculate masks
            positive_mask = (confidance_gt.view(-1) > 0)
            positive_mask_org_shape = (confidance_gt > 0)

            # exist confidance loss
            exist_condidence_loss = exist_condidence_loss + \
                                    torch.sum((1 - confidance[positive_mask_org_shape]) ** 2) / \
                                    torch.sum(positive_mask_org_shape)

            # non exist confidance loss
            target = confidance[confidance_gt == 0]
            nonexist_confidence_loss = nonexist_confidence_loss + \
                                       torch.sum((target[target > self.threshold_non_exist]) ** 2) / \
                                       (torch.sum(target > self.threshold_non_exist) + 1)

            # offset loss
            offset_x_gt = offset_gt[:, 0:1, :, :]
            offset_y_gt = offset_gt[:, 1:2, :, :]

            predict_x = offset[:, 0:1, :, :]
            predict_y = offset[:, 1:2, :, :]

            offset_loss = offset_loss + \
                          torch.sum((offset_x_gt[positive_mask_org_shape] - predict_x[positive_mask_org_shape]) ** 2) / \
                          torch.sum(positive_mask_org_shape) + \
                          torch.sum((offset_y_gt[positive_mask_org_shape] - predict_y[positive_mask_org_shape]) ** 2) / \
                          torch.sum(positive_mask_org_shape)

            # direction loss
            direct_loss += self.cal_masked_cls_loss(direct_pred,direct_gt,positive_mask,focal_gamma=1.0,focal_alpha=1.0) # 常规cross entropy

            # classification loss
            cls_loss += self.cal_masked_cls_loss(cls_pred,cls_gt,positive_mask,focal_gamma=1.0,focal_alpha=1.0) # 常规cross entropy

            #---------------------------------------------------#
            #  Clustering Loss
            #---------------------------------------------------#
            feature = feature.permute((0, 2, 3, 1)).contiguous()
            mean_feature_list_batches = []

            # Within Clusters Loss
            loss_variance = 0.0
            counter = 0
            for batch_idx in range(real_batch_size):
                instance_mask_gt_tmp = instance_mask_gt[batch_idx]
                one_batch_feature = feature[batch_idx]
                cluster_id = torch.unique(instance_mask_gt_tmp)[1:]

                mean_feature_list = []
                for tmp_id in cluster_id:
                    mask_tmp = (instance_mask_gt_tmp == tmp_id)
                    tmp_feature = one_batch_feature[mask_tmp]

                    if tmp_feature.shape[0] > self.cluster_min_num:
                        # calculate variance
                        mean_feature = torch.mean( tmp_feature, dim=0 )
                        mean_feature_list.append(mean_feature) # save features for calculating between cluster loss

                        distance_vector = tmp_feature - mean_feature
                        distance = torch.sqrt( torch.sum(distance_vector**2, dim=-1) )
                        # efficient_distance = torch.clamp_min( (distance - self.delta_v), 0)
                        efficient_distance = F.relu( distance - self.delta_v )

                        loss_instance = torch.mean( efficient_distance**2  )
                        loss_variance += loss_instance
                        counter+=1

                mean_feature_list_batches.append(mean_feature_list)

            sisc_loss += loss_variance / float(counter + 1e-8)

            # Between Clusters Loss
            loss_dist = 0.0
            counter = 0
            for batch_idx in range(len(mean_feature_list_batches)):
                mean_feature_list_ = mean_feature_list_batches[batch_idx]
                cluster_num = len(mean_feature_list_)

                for cluster_id_1 in range(0, cluster_num):
                    mean_feature_1 = mean_feature_list_[cluster_id_1]

                    for cluster_id_2 in range(cluster_id_1 + 1,cluster_num):
                        mean_feature_2 = mean_feature_list_[cluster_id_2]

                        # calculate variance
                        distance_vector = mean_feature_1 - mean_feature_2
                        distance = torch.sqrt( torch.sum(distance_vector * distance_vector ) )
                        # cluster_distance_neg = torch.clamp_min(  2 * self.delta_d- distance , 0)
                        cluster_distance_neg = F.relu( 2 * self.delta_d- distance )
                        loss_dist += cluster_distance_neg
                        counter+=1

            disc_loss += loss_dist / float(counter + 1e-8)

        center_map_loss = self.exist_condidence_loss * exist_condidence_loss + self.nonexist_confidence_loss * nonexist_confidence_loss
        cluster_loss = self.sisc_loss * sisc_loss + self.disc_loss * disc_loss
        return center_map_loss, offset_loss, cluster_loss, cls_loss, direct_loss


    def decode_result_vector(self,result):
        decoder_vector = []
        confidences, offsets, instances, directs, clss = result
        num_batch = confidences.shape[0]
        out_x = []
        out_y = []
        out_direct = []
        out_cls = []
        out_conf = []
        for batch_index in range(num_batch):
            confidence = confidences[batch_index].view(self.grid_y, self.grid_x).cpu().data.numpy()
            offset = offsets[batch_index].cpu().data.numpy()
            offset = np.rollaxis(offset, axis=2, start=0)
            offset = np.rollaxis(offset, axis=2, start=0)

            instance = instances[batch_index].cpu().data.numpy()
            instance = np.rollaxis(instance, axis=2, start=0)
            instance = np.rollaxis(instance, axis=2, start=0)

            direct = directs[batch_index].cpu().data.numpy()
            direct = np.rollaxis(direct, axis=2, start=0)
            direct = np.rollaxis(direct, axis=2, start=0)
            direct = np.argmax(direct,axis=-1)

            cls = clss[batch_index].cpu().data.numpy()
            cls = np.rollaxis(cls, axis=2, start=0)
            cls = np.rollaxis(cls, axis=2, start=0)
            cls = np.argmax(cls,axis=-1)

            # generate point and cluster
            raw_x, raw_y, direct_list, cls_list, conf_list = self.generate_result(confidence, offset, instance,direct,cls, self.thresh)

            # eliminate fewer points
            in_x, in_y, in_direct, in_cls, in_conf = self.eliminate_fewer_points(raw_x, raw_y,
                                                                                 direct_list, cls_list,
                                                                                 conf_list,
                                                                                 minal_pt_require=2)

            out_x.append(in_x)
            out_y.append(in_y)
            out_direct.append(in_direct)
            out_cls.append(in_cls)
            out_conf.append(in_conf)

        for idx in range(num_batch):
            vector_list = []
            out_x_ = out_x[idx]
            out_y_ = out_y[idx]
            out_direct_ = out_direct[idx]
            out_cls_ = out_cls[idx]
            out_conf_ = out_conf[idx]

            for (oneline_x, oneline_y, oneline_direct, oneline_cls, oneline_conf) in zip(out_x_,out_y_,out_direct_,out_cls_,out_conf_):
                vector_dict = dict()
                counter_list = []
                for cls_id in range(self.cls_num):
                    counter_list.append(oneline_cls.count(cls_id))
                max_id = np.argmax(np.array(counter_list))
                vector_dict["type_"] = max_id
                vector_dict["confidence_level"] = np.array(oneline_conf).mean()

                mask_id = np.where(oneline_cls == max_id)[0]

                #---------------------------------------------------#
                #  connect line by directions
                #---------------------------------------------------#
                pts_x = np.array(oneline_x).reshape(1, -1)
                pts_y = np.array(oneline_y).reshape(1, -1)
                pts = np.transpose(np.vstack([pts_x, pts_y]))[mask_id]
                direction = np.array(oneline_direct)[mask_id]

                if USE_DIRECTION_INFO:
                    pts = self.connect_line_by_direction(pts, direction,max_id)
                else:
                    pts = self.coarse_sample(pts, max_id)
                vector_dict["pts"] = pts
                vector_dict["pts_num"] = pts.shape[0]

                if pts.shape[0] > self.min_pt_num[max_id]:
                    vector_list.append(vector_dict)
            decoder_vector.append(vector_list)
        return decoder_vector

    def coarse_sample(self,pts, cls_id):
        dist_step = self.sample_distance[cls_id] / self.resolution
        pts_corse = []
        remove_flag = np.zeros(pts.shape[0],dtype=np.bool)
        for idx, pt in enumerate(pts):
            if not remove_flag[idx]:
                pts_corse.append(pt)
                delta_distance = np.sqrt( np.sum(np.abs(pts - pt)**2, axis=-1) )
                index_remove = np.where( delta_distance < dist_step)[0]
                remove_flag[index_remove] = True
        return np.array(pts_corse).reshape(-1,2)

    @staticmethod
    def judge_all_removed(pts_info_list):
        all_removed = True
        for idx, pt_info in enumerate(pts_info_list):
            if not pt_info["removed"]:
                all_removed = False
                return all_removed
        return all_removed

    def connect_one_direction(self, pt_rand_index, pts_info_list,direct_index, cls_id):
        dist_step = self.sample_distance[cls_id] / self.resolution
        pt_curr_index = pt_rand_index
        first_in = True
        vectorized_line = []

        while not self.judge_all_removed(pts_info_list):
            # 判断方向并标记方向为已经用
            if first_in:
                pd = pts_info_list[pt_curr_index]["direction"][direct_index]
                pts_info_list[pt_curr_index]["direction_indicator"][direct_index] = False
                first_in = False
            else:
                direct0 = pts_info_list[pt_curr_index]["direction_indicator"][0]
                direct1 = pts_info_list[pt_curr_index]["direction_indicator"][1]
                if direct0:
                    direct_index = 0
                elif direct1:
                    direct_index = 1
                else:
                    print("error occurs")
                    break

                pd = pts_info_list[pt_curr_index]["direction"][direct_index]
                pts_info_list[pt_curr_index]["direction_indicator"][direct_index] = False
                pts_info_list[pt_curr_index]["removed"] = True #

            angel = self.angle_list[pd]
            delta_x = dist_step * np.cos(angel / 180.0 * 3.141592)
            delta_y = -dist_step * np.sin(angel / 180.0 * 3.141592)

            # target point calculation
            pt_curr = pts_info_list[pt_curr_index]["point"]
            pt_target = np.array([pt_curr[0] + delta_x, pt_curr[1] + delta_y])

            # fileter points that are within radius
            for idx, pt_info in enumerate(pts_info_list):
                if not pt_info["removed"]:
                    pt_tmp = pt_info["point"]
                    dist_tmp = np.sqrt( np.sum( np.power(pt_curr - pt_tmp,2) ) )
                    if dist_tmp < dist_step:
                        pts_info_list[idx]["removed"] = True

            # find closest point
            dist_min = 100000
            pt_same_point_index = pt_curr_index
            for idx, pt_info in enumerate(pts_info_list):
                if not pt_info["removed"]:
                    pt_tmp = pt_info["point"]
                    dist_tmp = np.sqrt( np.sum( np.power(pt_target - pt_tmp,2) ) )
                    if (0 < dist_tmp < dist_min) and (pt_same_point_index != idx):
                        dist_min = dist_tmp
                        pt_curr_index = idx

            if dist_min > self.thershold_remove[cls_id] * dist_step:
                break

            pt_next = pts_info_list[pt_curr_index]["point"]
            # get direction
            theta = np.arctan2((-pt_curr[1] + pt_next[1]), (pt_curr[0] - pt_next[0] + 1e-8)) * 180.0 / np.pi # current - next
            if theta < 0:
                theta = 360 + theta

            pt_next_direct0 = pts_info_list[pt_curr_index]["direction"][0]
            pt_next_direct1 = pts_info_list[pt_curr_index]["direction"][1]

            # mark the connected direction of p next as taken
            delta0 = abs(theta - self.angle_list[pt_next_direct0])
            delta1 = abs(theta - self.angle_list[pt_next_direct1])

            vectorized_line.append(pt_next)
            if delta0 < delta1:
                pts_info_list[pt_curr_index]["direction_indicator"][0] = False
            else:
                pts_info_list[pt_curr_index]["direction_indicator"][1] = False

        return vectorized_line

    def connect_line_by_direction(self, pts, direction, cls_id):
        pts_info_list = []
        for (pt, direct) in zip(pts,direction):
            tmp_dict = dict()
            tmp_dict["point"] = pt
            tmp_dict["direction"] = [direct, int(direct + int(len(self.angle_list)/2)) %  len(self.angle_list) ]
            tmp_dict["direction_indicator"] = [True, True]
            tmp_dict["removed"] = False
            pts_info_list.append(tmp_dict)

        if DEBUG:
            # debug visualization
            img_vis = np.zeros([256,512,3],dtype=np.uint8)
            for pt in pts:
                cv2.circle(img_vis, (pt[0], pt[1]), 2, (0,255,0), thickness=-1)
            plt.imshow(img_vis)
            plt.show()

        # pt_list pt_rand taken_token direction
        # pt_rand_index = np.random.randint(pts.shape[0])
        pt_rand_index = int(pts.shape[0]/2)

        vectorized_line1 = self.connect_one_direction(pt_rand_index, pts_info_list, 0, cls_id)
        vectorized_line2 = self.connect_one_direction(pt_rand_index, pts_info_list, 1, cls_id)
        vectorized_line = vectorized_line2[::-1] + [pts_info_list[pt_rand_index]["point"]] + vectorized_line1
        vectorized_line = np.array(vectorized_line)

        if DEBUG:
            img_vis = np.zeros([256,512,3],dtype=np.uint8)
            for index in range(len(vectorized_line)-1):
                pt_x = int(vectorized_line[index][0])
                pt_y = int(vectorized_line[index][1])
                pt_x_ = int(vectorized_line[index+1][0])
                pt_y_ = int(vectorized_line[index+1][1])
                img_vis = cv2.line(img_vis,(pt_x, pt_y),(pt_x_,pt_y_), (0,255,0),2)
            plt.imshow(img_vis)
            plt.show()

        return vectorized_line # 有顺序的line

    def generate_result(self,confidance, offsets, instance, directs, clss, thresh):

        mask = confidance > thresh

        if DEBUG:
            plt.imshow(mask)
            plt.show()

        grid = self.grid_location[mask]

        confd = confidance[mask]
        offset = offsets[mask]
        feature = instance[mask]
        direct = directs[mask]
        cls = clss[mask]

        confd_list = []
        lane_feature = []
        x = []
        y = []
        direct_list = []
        cls_list = []


        if DEBUG:
            painter = np.zeros([256,512,3],dtype=np.uint8)
            for i in range(len(grid)):
                point_x = int((offset[i][0] + grid[i][0]) * self.resize_ratio)
                point_y = int((offset[i][1] + grid[i][1]) * self.resize_ratio)
                if point_x >= self.x_size or point_x < 0 or point_y >= self.y_size or point_y < 0:
                    continue
                else:
                    cv2.circle(painter, (point_x, point_y),1,(255,0,0),-1)
            plt.imshow(painter)
            plt.show()

        for i in range(len(grid)):
            point_x = int((offset[i][0] + grid[i][0]) * self.resize_ratio)
            point_y = int((offset[i][1] + grid[i][1]) * self.resize_ratio)
            if point_x >= self.x_size or point_x < 0 or point_y >= self.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                confd_list.append([confd[i]])
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
                direct_list.append([direct[i]])
                cls_list.append([cls[i]])

            else:
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j) ** 2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= self.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index] * len(x[min_feature_index]) +
                                                       feature[i]) / (len(x[min_feature_index]) + 1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)

                    direct_list[min_feature_index].append(direct[i])
                    cls_list[min_feature_index].append(cls[i])
                    confd_list[min_feature_index].append(confd[i])

                elif min_feature_dis > self.threshold_instance * 2:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                    direct_list.append([direct[i]])
                    cls_list.append([cls[i]])
                    confd_list.append([confd[i]])

        return x, y, direct_list, cls_list, confd_list

    @staticmethod
    def eliminate_fewer_points(x, y, direct_list, cls_list, conf_list, minal_pt_require = 4):
        # eliminate fewer points
        out_x = []
        out_y = []
        out_direct = []
        out_cls = []
        out_conf = []
        for i, j, direct, cls, conf in zip(x, y, direct_list, cls_list,conf_list):
            if len(i) > minal_pt_require:
                out_x.append(i)
                out_y.append(j)
                out_direct.append(direct)
                out_cls.append(cls)
                out_conf.append(conf)
        return out_x, out_y, out_direct, out_cls, out_conf

    @staticmethod
    def sort_along_y(x, y, direct, cls ,conf):
        out_x = []
        out_y = []
        out_direct = []
        out_cls = []
        out_conf = []

        for i, j, direc, cl, cf in zip(x, y, direct, cls, conf):
            i = np.array(i)
            j = np.array(j)
            direc = np.array(direc)
            cl = np.array(cl)
            cf = np.array(cf)

            ind = np.argsort(j, axis=0)
            out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
            out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
            out_direct.append(np.take_along_axis(direc, ind[::-1], axis=0).tolist())
            out_cls.append(np.take_along_axis(cl, ind[::-1], axis=0).tolist())
            out_conf.append(np.take_along_axis(cf, ind[::-1], axis=0).tolist())

        return out_x, out_y, out_direct, out_cls, out_conf

    @staticmethod
    def mask_for_lines(lines, mask, idx, thickness):  # fix bug
        coords = np.asarray(lines, np.int32)
        coords = coords.reshape((-1, 2))
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        # cv2.polylines(mask, [coords], False, color=idx * 20, thickness=thickness)
        idx += 1
        return mask, idx


    def draw_points(self,in_vector, image):
        # "pts" , "pts_num" , "type_" ,"confidence_level"
        color_map = np.random.randint(100, 256, (100, 3))
        for idx, one_vector in enumerate(in_vector):
            pts = one_vector["pts"]
            type_ = one_vector["type_"]

            if not DRAW_LINE:
                for index in range(len(pts)):
                    pt_x = int(pts[index][0])
                    pt_y = int(pts[index][1])
                    image = cv2.circle(image, (pt_x, pt_y), 3, (int(color_map[idx,0]),int(color_map[idx,1]),int(color_map[idx,2])), -1)
                    # image = cv2.circle(image, (pt_x, pt_y), 3, self.color[ int(type_) ], -1)

            else:
                for index in range(len(pts)-1):
                    pt_x = int(pts[index][0])
                    pt_y = int(pts[index][1])

                    pt_x_ = int(pts[index+1][0])
                    pt_y_ = int(pts[index+1][1])

                    image = cv2.line(image, (pt_x, pt_y), (pt_x_, pt_y_), self.color[int(type_)], 3)
                    if index ==0:
                        image = cv2.circle(image, (pt_x, pt_y), 4, self.color[ int(type_) ], -1)
                    if index == len(pts) - 2:
                        image = cv2.circle(image, (pt_x_, pt_y_), 4, self.color[int(type_)], -1)

        return image


    def display(self, images, decoded_vector_list):
        num_batch = len(images)
        out_images = images
        for i in range(num_batch):
            in_vector = decoded_vector_list[i]
            out_images[i] = self.draw_points(in_vector, images[i])
        return out_images


def print_val_info(val_info, CHANNEL_NAMES):
    for key, value in val_info.items():
        print()
        print("=== %s ===" % key)
        for idx, cls_name in enumerate(CHANNEL_NAMES):
            print(cls_name + ":")
            print(value.numpy()[idx])
        print()

#---------------------------------------------------#
#  TEST Module
#---------------------------------------------------#
if __name__ == '__main__':
    #---------------------------------------------------#
    #  Parameters setting
    #---------------------------------------------------#
    import yaml, time
    CFG_PATH = "../cfgs/ultrabev_stn_seq3.yml"
    BATCH_SIZE = 2
    ITERATION_NUM = 5

    #---------------------------------------------------#
    #  Parameters Loading
    #---------------------------------------------------#
    cfgs = yaml.safe_load(open(CFG_PATH))
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    x_bound = cfgs["dataloader"]["x_bound"]
    y_bound = cfgs["dataloader"]["y_bound"]

    channels = cfgs["backbone"]["channels"]
    feat_ratio = cfgs["backbone"]["bev_feat_ratio"]
    bev_width = int((x_bound[1] - x_bound[0]) / (x_bound[2]))
    bev_height = int((y_bound[1] - y_bound[0]) / (y_bound[2]))
    bev_feat_width = int(bev_width / feat_ratio)
    bev_feat_height = int(bev_height / feat_ratio)

    #---------------------------------------------------#
    #  Vector Parameters Loading
    #---------------------------------------------------#
    vector_scale_ratio = cfgs["vector"]["vector_scale_ratio"]
    cluster_feat_dim = cfgs["vector"]["cluster_feat_dim"]
    exist_condidence_loss = cfgs["vector"]["exist_condidence_loss"]
    nonexist_confidence_loss = cfgs["vector"]["nonexist_confidence_loss"]
    sisc_loss = cfgs["vector"]["sisc_loss"]
    disc_loss = cfgs["vector"]["disc_loss"]
    delta_v = cfgs["vector"]["delta_v"]
    delta_d = cfgs["vector"]["delta_d"]
    thresh = cfgs["vector"]["thresh"]
    threshold_instance = cfgs["vector"]["threshold_instance"]
    class_num = cfgs["vector"]["class_num"]
    threshold_non_exist = cfgs["vector"]["threshold_non_exist"]
    sample_distance = cfgs["vector"]["sample_distance"]
    resolution = cfgs["vector"]["resolution"]
    threshold_remove = cfgs["vector"]["threshold_remove"]
    angle_list = cfgs["vector"]["angle_list"]
    cluster_min_num = cfgs["vector"]["cluster_min_num"]

    # upsample time
    upsample_time = int(np.log2(feat_ratio / vector_scale_ratio))
    use_half_direct = cfgs["vector"]["use_half_direct"]
    use_resize_conv_os = cfgs["vector"]["use_resize_conv_os"]

    vector_header = VectorHeader(
                                channels,
                                cluster_feat_dim,
                                exist_condidence_loss = exist_condidence_loss,
                                nonexist_confidence_loss = nonexist_confidence_loss,
                                sisc_loss=sisc_loss,
                                disc_loss=disc_loss,
                                grid_x=bev_feat_width,
                                grid_y=bev_feat_height,
                                x_size=net_input_width,
                                y_size=net_input_height,
                                delta_v=delta_v,
                                delta_d=delta_d,
                                thresh=thresh,
                                threshold_instance=threshold_instance,
                                resize_ratio=feat_ratio,
                                class_num=class_num,
                                threshold_non_exist=threshold_non_exist,
                                sample_distance=sample_distance,
                                resolution=resolution,
                                threshold_remove=threshold_remove,
                                angle_list=angle_list,
                                up_sample_time=upsample_time,
                                cluster_min_num=cluster_min_num,
                                use_half_direct=use_half_direct,
                                use_resize_conv_os=use_resize_conv_os,
    ).cuda()

    #---------------------------------------------------#
    #  Inference Test
    #---------------------------------------------------#
    dummy_input = torch.randn((BATCH_SIZE, channels, bev_feat_height, bev_feat_width)).cuda()
    outs = vector_header(dummy_input)
    print(outs[0].shape)
    print(outs[1].shape)
    print(outs[2].shape)
    print(outs[3].shape)
    print(outs[4].shape)

    avg_runtime = 0.0
    for _ in range(50):
        tic = time.time()
        ouptut = vector_header(dummy_input)
        torch.cuda.synchronize()
        print("inference time: %i" %(1000*(time.time() - tic)))
        avg_runtime += 1000*(time.time() - tic)
    print("average time: %i" % (avg_runtime/50))

    #---------------------------------------------------#
    #  Loss Calculation Test
    #---------------------------------------------------#
    confidance_gt = torch.tensor(np.load("sample/gt_binary.npy")).cuda()
    offset_gt = torch.tensor(np.load("sample/gt_offsetmap.npy")).cuda()
    instance_mask_gt = torch.tensor(np.load("sample/gt_instancemap.npy")).cuda()

    if use_half_direct:
        direct_gt = torch.tensor(np.load("sample/gt_point_direction_half.npy")).cuda()

    else:
        direct_gt = torch.tensor(np.load("sample/gt_point_direction_full.npy")).cuda()

    cls_gt = torch.tensor(np.load("sample/gt_classmap.npy")).cuda()

    center_map_loss, offset_loss, cluster_loss, cls_loss, direct_loss = \
        vector_header.cal_loss_vector(confidance_gt, offset_gt, instance_mask_gt, direct_gt, cls_gt, outs)

    print("center_map_loss: %f" % center_map_loss)
    print("offset_loss: %f" %  offset_loss)
    print("cluster_loss: %f" % cluster_loss)
    print("cls_loss: %f" % cls_loss)
    print("direct_loss: %f" % direct_loss)

    #---------------------------------------------------#
    #  Demo Test
    #---------------------------------------------------#
    dummy_input_demp = torch.randn((1, channels, bev_feat_height, bev_feat_width )).cuda()
    outs_demo = vector_header(dummy_input_demp)

    # Decode Result
    decoded_vector_list = vector_header.decode_result_vector(outs_demo)

    # Display
    map_mask_vis = np.zeros((bev_height, bev_width, 3), np.uint8)
    map_mask_vis = vector_header.display([map_mask_vis], decoded_vector_list)

    #---------------------------------------------------#
    #  Validation Test
    #---------------------------------------------------#
    from bev_vector.devkit.evaluation.chamfer_distance import semantic_mask_chamfer_dist_cum
    from bev_vector.devkit.evaluation.AP import instance_mask_AP
    from bev_vector.devkit.evaluation.iou import get_batch_iou

    SAMPLED_RECALLS = torch.linspace(0.1, 1, cfgs["vector"]["sampled_recalls_num"])
    CHANNEL_NAMES = cfgs["vector"]["class_list"]
    THRESHOLDS = cfgs["vector"]["thresholds"]
    CD_threshold = cfgs["vector"]["cd_threshold"]
    resolution_x = cfgs["vector"]["resolution_x"]
    resolution_y = cfgs["vector"]["resolution_y"]
    max_instance_num = cfgs["vector"]["max_instance_num"]
    thickness = cfgs["vector"]["thickness"]

    max_channel = len(CHANNEL_NAMES)
    total_CD1 = torch.zeros(max_channel)
    total_CD2 = torch.zeros(max_channel)
    total_CD_num1 = torch.zeros(max_channel)
    total_CD_num2 = torch.zeros(max_channel)
    total_intersect = torch.zeros(max_channel)
    total_union = torch.zeros(max_channel)
    AP_matrix = torch.zeros((max_channel, len(THRESHOLDS)))
    AP_count_matrix = torch.zeros((max_channel, len(THRESHOLDS)))

    vector_sample = ["/data/zdx/Data/data_nuscene/0_19/14/gt_label/vector_label.npy"] * BATCH_SIZE
    vector_list_all = [vector_sample] * ITERATION_NUM

    for i in range(ITERATION_NUM):
        dummy_input_demp = torch.randn((BATCH_SIZE, channels, bev_feat_height, bev_feat_width)).cuda()
        outs= vector_header(dummy_input_demp)
        decoded_vector_list = vector_header.decode_result_vector(outs)

        #---------------------------------------------------#
        #  generate prediction map and confidence list
        #---------------------------------------------------#
        masks_batches = list()
        confidence_level_list = list()
        for one_sampel_vector_list in decoded_vector_list:
            confidence_level_ = [-1]
            decoded_vector_dict = {"0": [], "1": [], "2": []}
            decoded_vector_confidence = {"0": [], "1": [], "2": []}

            for one_vector in one_sampel_vector_list:
                type_ = str(one_vector["type_"])
                decoded_vector_dict[type_].append(one_vector["pts"])
                decoded_vector_confidence[type_].append(one_vector["confidence_level"])

            # convert to pred map
            idx = 1
            masks = []
            for key, value_container in decoded_vector_dict.items():
                map_mask = np.zeros((bev_height, bev_width), np.uint8)
                confidence_container = decoded_vector_confidence[key]
                for one_line,confidence in zip(value_container,confidence_container):
                    map_mask, idx = vector_header.mask_for_lines(one_line, map_mask, idx, thickness)  #
                    confidence_level_.append(confidence)
                masks.append(map_mask)

            masks_one_sample = torch.tensor(np.stack(masks))
            masks_batches.append(masks_one_sample)

            confidence_level = torch.tensor(confidence_level_ + [-1] * (max_instance_num - len(confidence_level_)))
            confidence_level_list.append(confidence_level)

        pred_map = torch.stack(masks_batches)
        confidence_level = torch.stack(confidence_level_list)

        #---------------------------------------------------#
        #  generate gt map list
        #---------------------------------------------------#
        gt_map = []
        vector_list_tmp = vector_list_all[i]
        for tmp_file_name in vector_list_tmp:
            vector_label = np.load(tmp_file_name, allow_pickle=True).item()
            # convert to gt map

            idx = 1
            thickness = 1
            masks = []
            for key, value_container in vector_label.items():
                map_mask = np.zeros((bev_height, bev_width), np.uint8)
                for one_line in value_container:
                    map_mask, idx = vector_header.mask_for_lines(one_line, map_mask, idx, thickness)  #

                masks.append(map_mask)

            masks_one_sample = torch.tensor(np.stack(masks))
            gt_map.append(masks_one_sample)
        gt_map = torch.stack(gt_map)

        # gt map && confidences && predict map calculation
        intersect, union = get_batch_iou(pred_map, gt_map)
        CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(pred_map, gt_map, resolution_x, resolution_y, threshold=CD_threshold)

        instance_mask_AP(AP_matrix, AP_count_matrix, pred_map, gt_map, resolution_x, resolution_y,
                         confidence_level, THRESHOLDS, sampled_recalls=SAMPLED_RECALLS)

        total_intersect += intersect
        total_union += union
        total_CD1 += CD1
        total_CD2 += CD2
        total_CD_num1 += num1
        total_CD_num2 += num2

    CD_pred = total_CD1 / total_CD_num1
    CD_label = total_CD2 / total_CD_num2
    CD = (total_CD1 + total_CD2) / (total_CD_num1 + total_CD_num2)
    CD_pred[CD_pred > CD_threshold] = CD_threshold
    CD_label[CD_label > CD_threshold] = CD_threshold
    CD[CD > CD_threshold] = CD_threshold

    dict_info =  {
        'iou': total_intersect / total_union,
        'CD_pred': CD_pred,
        'CD_label': CD_label,
        'CD': CD,
        'Average_precision': AP_matrix / AP_count_matrix,
    }

    # print
    print_val_info(dict_info, CHANNEL_NAMES)



