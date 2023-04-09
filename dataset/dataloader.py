"""
Function: Dataloader For VectorSpace Model
Author: Zhan Dong Xu
Date: 2021/11/11
"""

from abc import ABC

import os
import numpy as np
import yaml, math
from pyquaternion import Quaternion

import torch
from torch.utils.data import Dataset
import torch.utils.data.dataloader
from dataset.utility import get_img_whc, imread, create_subset, resize_by_wh, bgr2rgb, imagenet_normalize
from mmcv.runner import BaseModule


class SinePositionalEncoding(BaseModule, ABC):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


#---------------------------------------------------#
#  BEV数据增强
#---------------------------------------------------#
import imgaug as ia
import imgaug.augmenters as iaa

#---------------------------------------------------#
# bev图像小幅度增强 -- 这里增强幅度不能大，而且不能翻转，
# 只能以微扰的方式进行图像的几何变换，一定要保持一个像素在一个
# 小范围内变动，否则基于stn和transformer的方式都会有问题
# 这个数据增强，其实本质是模拟相机的抖动
#---------------------------------------------------#
def bev_image_aug(image,
                  GaussianBlur = (0.5, 1.5),
                  LinearContrast = (0.8, 1.2),
                  Multiply = (0.8, 1.2),
                  AdditiveGaussianNoise = 0.1,
                  WithColorspace_Multiply_1 = (0.7, 1.3),
                  WithColorspace_Multiply_2 = (0.1, 2.0),
                  WithColorspace_Multiply_3 = (0.5, 1.5),
                  TranslateX = (-5, 5),
                  TranslateY = (-5, 5),
                  ShearX = (-5, 5),
                  ShearY = (-5, 5),
                  Rotate = (-3, 3),
                  Crop = 0.05,
                  aug_probability = 0.6,
                  ):
    #---------------------------------------------------#
    #  定义增强序列
    #---------------------------------------------------#

    # 颜色增强
    color_shift = iaa.OneOf([
        iaa.GaussianBlur(sigma=GaussianBlur),
        iaa.LinearContrast(LinearContrast, per_channel=False),
        iaa.Multiply(Multiply, per_channel=0.2),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, AdditiveGaussianNoise * 255), per_channel=0.5),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(0, iaa.Multiply(WithColorspace_Multiply_1))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(1, iaa.Multiply(WithColorspace_Multiply_2))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(2, iaa.Multiply(WithColorspace_Multiply_3))),
    ])

    # 几何增强
    posion_shift = iaa.SomeOf(6, [ iaa.TranslateX(px=TranslateX),
                                   iaa.ShearX(shear=ShearX),
                                   iaa.TranslateY(px=TranslateY),
                                   iaa.ShearY(shear=ShearY),
                                   iaa.Rotate(rotate=Rotate),
                                   iaa.Crop(percent=([0, Crop], [0, Crop], [0, Crop], [0, Crop]), keep_size=True)])

    aug = iaa.Sequential([
        iaa.Sometimes(p=aug_probability, then_list=color_shift),
        iaa.Sometimes(p=aug_probability, then_list=posion_shift)], random_order=True)

    # =========================================
    # 开始数据增强
    # =========================================
    args = dict(images=[image])
    batch = ia.Batch(**args)
    batch_aug = list(aug.augment_batches([batch]))[0]  # augment_batches returns a generator
    image_aug = batch_aug.images_aug[0]
    return image_aug

class BEVData(Dataset):
    def __init__(self, cfgs, mode):
        super().__init__()
        self.cfgs = cfgs
        self.mode = mode

        # load sequential parameters
        self.seq_num = self.cfgs["dataloader"]["seq_num"]

        #---------------------------------------------------#
        #  Load Universal Parameters
        #---------------------------------------------------#
        # data set list
        self.data_list = self.cfgs["dataloader"]["data_list"]
        self.data_list_train = os.path.join(self.data_list, "train_seq%i.txt" %self.seq_num)
        self.data_list_valid = os.path.join(self.data_list, "valid_seq%i.txt" %self.seq_num)

        # multiview features fusion strategies
        # True: features => transformer encoder => transformer decoder => fused bev features
        # False: features => spatial transformation layers => transformer encoder => fused bev features
        self.use_transformer_decoder = self.cfgs["backbone"]["use_transformer_decoder"]

        # load size parameters
        self.network_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.network_input_height = self.cfgs["dataloader"]["network_input_height"]
        self.cam_num = self.cfgs["dataloader"]["cam_num"]
        self.cam_num = self.cfgs["dataloader"]["cam_num"]
        self.layer_id = self.cfgs["backbone"]["layer_id"]
        self.layers_dim = self.cfgs["backbone"]["dim_array"]
        self.cam_list = self.cfgs["dataloader"]["camera_list"]
        self.x_bound = self.cfgs["dataloader"]["x_bound"]
        self.y_bound = self.cfgs["dataloader"]["y_bound"]
        self.bev_feat_ratio = self.cfgs["backbone"]["bev_feat_ratio"]



        # image size
        self.feat_height = int(self.network_input_height / pow(2, self.layer_id + 1))
        self.feat_width = int(self.network_input_width / pow(2, self.layer_id + 1))
        self.bev_width = int((self.x_bound[1] - self.x_bound[0]) / (self.x_bound[2]))
        self.bev_height = int((self.y_bound[1] - self.y_bound[0]) / (self.y_bound[2]))
        self.bev_feat_width = int(self.bev_width / self.bev_feat_ratio)
        self.bev_feat_height = int(self.bev_height / self.bev_feat_ratio)
        self.embed_dim = int(self.layers_dim[self.layer_id - 1] * self.cam_num / 2)

        # augment images
        self.with_aug = self.cfgs["dataloader"]["with_aug"]
        if self.mode == "val": self.with_aug = False
        self.GaussianBlur = tuple(self.cfgs["dataloader"]["GaussianBlur"])
        self.Multiply = tuple(self.cfgs["dataloader"]["Multiply"])
        self.LinearContrast = tuple(self.cfgs["dataloader"]["LinearContrast"])
        self.AdditiveGaussianNoise = self.cfgs["dataloader"]["AdditiveGaussianNoise"]
        self.WithColorspace_Multiply_1 = tuple(self.cfgs["dataloader"]["WithColorspace_Multiply_1"])
        self.WithColorspace_Multiply_2 = tuple(self.cfgs["dataloader"]["WithColorspace_Multiply_2"])
        self.WithColorspace_Multiply_3 = tuple(self.cfgs["dataloader"]["WithColorspace_Multiply_3"])
        self.TranslateX = tuple(self.cfgs["dataloader"]["TranslateX"])
        self.TranslateY = tuple(self.cfgs["dataloader"]["TranslateY"])
        self.ShearX = tuple(self.cfgs["dataloader"]["ShearX"])
        self.ShearY = tuple(self.cfgs["dataloader"]["ShearY"])
        self.Rotate = tuple(self.cfgs["dataloader"]["Rotate"])
        self.Crop = self.cfgs["dataloader"]["Crop"]
        self.aug_probability = self.cfgs["dataloader"]["aug_probability"]

        # detection header
        self.det_class_list = self.cfgs["detection"]["class_list"]
        self.det_num_classes = len(self.det_class_list)
        self.det_class_to_ind = dict(zip(self.det_class_list, range(self.det_num_classes)))

        # vector header
        self.seg_class_list = self.cfgs["vector"]["class_list"]
        self.vector_scale_ratio = self.cfgs["vector"]["vector_scale_ratio"]
        self.angle_list = self.cfgs["vector"]["angle_list"]
        self.full_direction_num = len(self.angle_list)
        self.cls_index_list = list(range(self.full_direction_num))
        self.half_circle_num = int(len(self.angle_list)/2)
        self.vector_feat_width = int(self.bev_width / self.vector_scale_ratio)
        self.vector_feat_height = int(self.bev_height / self.vector_scale_ratio)
        self.use_half_direct = self.cfgs["vector"]["use_half_direct"]

        dataset_pairs = dict(
            train=create_subset(self.data_list_train, self.cam_list),
            val=create_subset(self.data_list_valid, self.cam_list)
        )

        if self.mode not in dataset_pairs.keys():
            raise NotImplementedError(f'mode should be one of {dataset_pairs.keys()}')
        self.image_annot_path_pairs = dataset_pairs.get(self.mode)


        # Collect Function
        self.collate_fn = Collater(target_height=self.network_input_height,
                                   target_width=self.network_input_width,
                                   embedding_dim=self.embed_dim)


    def __len__(self):
        """Get the length.

        :return: the length of the returned iterators.
        :rtype: int
        """
        return len(self.image_annot_path_pairs)

    def __getitem__(self, idx):
        """Get an item of the dataset according to the index.

        :param idx: index
        :type idx: int
        :return: an item of the dataset according to the index
        :rtype: dict
        """
        return self.prepare_img(idx)


    def prepare_img(self, idx):
        """Prepare an image for training.

        :param idx:index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        target_pair = self.image_annot_path_pairs[idx]

        imgae_folder_name = os.path.basename(os.path.dirname(os.path.dirname(target_pair['image_path'][-1][0])))

        image_seq_list = list()
        for outer in range(self.seq_num):
            image_arr_list = list()
            for i in range(self.cam_num):
                image_arr = imread(target_pair['image_path'][outer][i])

                if self.with_aug:
                    image_arr = bev_image_aug(image_arr,
                                              self.GaussianBlur,
                                              self.LinearContrast,
                                              self.Multiply,
                                              self.AdditiveGaussianNoise,
                                              self.WithColorspace_Multiply_1,
                                              self.WithColorspace_Multiply_2,
                                              self.WithColorspace_Multiply_3,
                                              self.TranslateX,
                                              self.TranslateY,
                                              self.ShearX,
                                              self.ShearY,
                                              self.Rotate,
                                              self.Crop,
                                              self.aug_probability)

                image_arr_list.append(image_arr)
            image_seq_list.append(image_arr_list)

        whc = get_img_whc(image_seq_list[-1][0])

        # ---------------------------------------------------#
        #  Vector Label Generation
        # ---------------------------------------------------#
        vector_label = np.load(target_pair['annot_path_seg'], allow_pickle=True).item()
        gt_binary, gt_offsetmap, gt_instancemap, gt_classmap, gt_point_direction = self.transfer_vector_to_gtmap(vector_label)

        # ---------------------------------------------------#
        #  Detection Label Generation
        # ---------------------------------------------------#
        obj_annotation, obj_labels = self.load_detect_annot(target_pair['annot_path_detect'])

        #---------------------------------------------------#
        #  Normalize Images
        #---------------------------------------------------#
        network_input_images = list()
        for image_arr_list in image_seq_list:
            network_input_one_seq_images = list()
            for image_arr in image_arr_list:
                network_input_image = bgr2rgb(resize_by_wh(img=image_arr,
                                                           width= self.network_input_width,
                                                           height=self.network_input_height))

                network_input_image = np.transpose(imagenet_normalize(img=network_input_image), (2, 0, 1)).astype('float32')
                network_input_one_seq_images.append(network_input_image)
            network_input_images.append(network_input_one_seq_images)

        #---------------------------------------------------#
        #  Generate Ego Motion Info **** IMPORTANT
        #---------------------------------------------------#
        ego_motion_path_list = target_pair['annot_ego']
        ego_moion_info_list = list()
        for idx in range(len(ego_motion_path_list) - 1):
            ego_info_curr = yaml.safe_load(open(ego_motion_path_list[idx],"r"))
            ego_info_next = yaml.safe_load(open(ego_motion_path_list[idx + 1],"r"))
            ego_dict_curr = self.get_ego_pose_dict(ego_info_curr)
            ego_dict_next = self.get_ego_pose_dict(ego_info_next)
            egopose_t0 = self.convert_egopose_to_matrix_numpy(ego_dict_curr)
            egopose_t1 = self.convert_egopose_to_matrix_numpy(ego_dict_next)
            future_egomotion = self.invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
            future_egomotion[3, :3] = 0.0
            future_egomotion[3, 3] = 1.0
            future_egomotion = torch.Tensor(future_egomotion).float()
            # Convert to 6DoF vector
            future_egomotion = self.mat2pose_vec(future_egomotion)
            ego_moion_info_list.append(future_egomotion)

        future_egomotion_target = torch.Tensor(np.eye(4, dtype=np.float32)).float()
        future_egomotion_target = self.mat2pose_vec(future_egomotion_target)
        ego_moion_info_list.append(future_egomotion_target) # 当前帧的ego motion为单位矩阵

        #---------------------------------------------------#
        #  Generate Meta Info
        #---------------------------------------------------#
        img_meta = dict()
        img_meta["filename"] = target_pair['image_path'][-1]
        img_meta["ori_shape"] = (self.bev_height, self.bev_width, 3)
        img_meta["img_shape"] = (self.bev_height, self.bev_width, 3)
        img_meta["pad_shape"] = (self.bev_height, self.bev_width, 3)
        img_meta["scale_factor"] = [1., 1., 1., 1.]
        img_meta["flip"] = False
        img_meta["flip_direction"] = None
        img_meta["img_norm_cfg"] = {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                                    'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                                    'to_rgb': True}

        img_meta["border"] = [0, self.bev_height, 0, self.bev_width]
        img_meta["batch_input_shape"] = (self.bev_height, self.bev_width)
        img_metas = [img_meta]

        #---------------------------------------------------#
        #  PM Matrix
        #---------------------------------------------------#
        # perspective matrix from bev to perspective view
        pm_matrix_list = []
        for pm_matrix_path in target_pair['annot_pm']:
            pm_matrix = np.load(pm_matrix_path)
            pm_matrix_list.append(pm_matrix)
        pm_matrix_list = np.stack(pm_matrix_list, axis=0)

        result = dict(
                      image=network_input_images,
                      src_image_shape=whc,
                      src_image_path=target_pair['image_path'][-1],
                      gt_binary=gt_binary,                      # vector label
                      gt_offsetmap=gt_offsetmap,                # vector label
                      gt_instancemap=gt_instancemap,            # vector label
                      gt_classmap=gt_classmap,                  # vector label
                      gt_point_direction=gt_point_direction,    # vector label
                      gt_det_box=obj_annotation,
                      gt_det_label=obj_labels,
                      meta_data=img_metas,
                      pm_matrix=pm_matrix_list,
                      img_name_list=imgae_folder_name,
                      vector_label_info=vector_label,
                      ego_motion_info=ego_moion_info_list,
                      )

        return result

    #---------------------------------------------------#
    #  Detection Header Function
    #---------------------------------------------------#
    def load_detect_annot(self, labels_txt):
        annotations_list = open(labels_txt).readlines()
        annotations = np.zeros((0, 9))

        # some images appear to miss annotations
        if len(annotations_list) == 0:
            annotations_bboxes = annotations[:, 0:8]
            annotations_label = annotations[:, 8]
            return annotations_bboxes, annotations_label

        for idx, one_label in enumerate(annotations_list):
            one_label = one_label.strip("\n").split(",")
            category_id = self.det_class_list.index(one_label[0]) - 1 # 这里减一为去掉背景
            x1 = int(one_label[1])
            y1 = int(one_label[2])
            x2 = int(one_label[3])
            y2 = int(one_label[4])
            x3 = int(one_label[5])
            y3 = int(one_label[6])
            x4 = int(one_label[7])
            y4 = int(one_label[8])

            # counter clockwise 90 degree
            x1_ = y1
            y1_ = self.bev_height - x1
            flag_in_range1_ = (0<=x1_<self.bev_width-1) and (0<=y1_<self.bev_height-1)

            x2_ = y2
            y2_ = self.bev_height - x2
            flag_in_range2_ = (0<=x2_<self.bev_width-1) and (0<=y2_<self.bev_height-1)

            x3_ = y3
            y3_ = self.bev_height - x3
            flag_in_range3_ = (0<=x3_<self.bev_width-1) and (0<=y3_<self.bev_height-1)

            x4_ = y4
            y4_ = self.bev_height - x4
            flag_in_range4_ = (0<=x4_<self.bev_width-1) and (0<=y4_<self.bev_height-1)

            flag_merge = flag_in_range1_ and flag_in_range2_ and flag_in_range3_ and flag_in_range4_
            if not flag_merge:
                continue

            annotation = np.zeros((1, 9))
            annotation[0, :8] = [x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_]
            annotation[0, 8] = category_id
            annotations = np.append(annotations, annotation, axis=0)

        annotations_bboxes = annotations[:,0:8]
        annotations_label = annotations[:,8]
        return annotations_bboxes, annotations_label

    #---------------------------------------------------#
    #  Vector Header Function
    #---------------------------------------------------#
    def generate_direction(self,point_index, lane):
        if point_index == 0:
            p2 = lane[1]
            p1 = lane[0]
            theta1 = np.arctan2((-p2[1] + p1[1]), (p2[0] - p1[0] + 1e-8)) * 180.0 / np.pi
            if theta1 < 0:
                theta1 = 360 + theta1

            minimal_diff_index = np.argmin(np.abs(np.array(self.angle_list) - theta1))
            direction1 = self.cls_index_list[minimal_diff_index]
            direction2 = int(direction1 + self.half_circle_num) % int(len(self.cls_index_list))

        elif point_index == len(lane) - 1:
            p2 = lane[-1]
            p1 = lane[-2]
            theta1 = np.arctan2((-p2[1] + p1[1]), (p2[0] - p1[0] + 1e-8)) * 180.0 / np.pi
            if theta1 < 0:
                theta1 = 360 + theta1

            minimal_diff_index = np.argmin(np.abs(np.array(self.angle_list) - theta1))
            direction1 = self.cls_index_list[minimal_diff_index]
            direction2 = int(direction1 + self.half_circle_num) % int(len(self.cls_index_list))

        else:
            p2 = lane[point_index + 1]
            p1 = lane[point_index]
            p3 = lane[point_index - 1]

            theta1 = np.arctan2((-p2[1] + p1[1]), (p2[0] - p1[0] + 1e-8)) * 180.0 / np.pi
            theta2 = np.arctan2((-p1[1] + p3[1]), (p1[0] - p3[0] + 1e-8)) * 180.0 / np.pi
            if theta1 < 0:
                theta1 = 360 + theta1
            if theta2 < 0:
                theta2 = 360 + theta2

            theta_avg = (theta1 + theta2) / 2
            minimal_diff_index = np.argmin(np.abs(np.array(self.angle_list) - theta_avg))
            direction1 = self.cls_index_list[minimal_diff_index]
            direction2 = int(direction1 + self.half_circle_num) % len(self.cls_index_list)

        return direction1, direction2


    def transfer_vector_to_gtmap(self, vector_label):
        gt_binary = np.zeros((1, self.vector_feat_height, self.vector_feat_width))
        gt_offsetmap = np.zeros((2, self.vector_feat_height, self.vector_feat_width))
        gt_instancemap = np.zeros((self.vector_feat_height, self.vector_feat_width))
        gt_classmap = np.zeros((len(self.seg_class_list),self.vector_feat_height, self.vector_feat_width))

        if self.use_half_direct:
            gt_point_direction = np.zeros(( self.half_circle_num,self.vector_feat_height, self.vector_feat_width))

        else:
            gt_point_direction = np.zeros(( self.full_direction_num,self.vector_feat_height, self.vector_feat_width))

        instance_idx = 0
        for cls_idx, (key, one_batch) in enumerate(vector_label.items()):
            for lane_index, lane in enumerate(one_batch):
                instance_idx+=1
                for point_index, point in enumerate(lane):
                    point_x = point[0]
                    point_y = point[1]
                    x_index = int(point_x / self.vector_scale_ratio)
                    y_index = int(point_y / self.vector_scale_ratio)

                    if x_index < self.vector_feat_width and y_index < self.vector_feat_height:
                        gt_binary[0][y_index][x_index] = 1.0
                        gt_offsetmap[0][y_index][x_index] = (point_x * 1.0 / self.vector_scale_ratio) - x_index
                        gt_offsetmap[1][y_index][x_index] = (point_y * 1.0 / self.vector_scale_ratio) - y_index
                        gt_instancemap[y_index][x_index] = instance_idx
                        gt_classmap[cls_idx][y_index][x_index] = 1.0

                        # direction calculation
                        direction1, direction2 = self.generate_direction(point_index, lane)

                        if self.use_half_direct:
                            gt_point_direction[min(direction1,direction2)][y_index][x_index] = 1.0

                        else:
                            gt_point_direction[direction1][y_index][x_index] = 1.0
                            gt_point_direction[direction2][y_index][x_index] = 1.0

        return gt_binary, gt_offsetmap, gt_instancemap, gt_classmap, gt_point_direction

    #---------------------------------------------------#
    #  sequential related
    #---------------------------------------------------#
    @staticmethod
    def convert_egopose_to_matrix_numpy(egopose):
        transformation_matrix = np.zeros((4, 4), dtype=np.float32)
        rotation = Quaternion(egopose['rotation']).rotation_matrix
        translation = np.array(egopose['translation'])
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation
        transformation_matrix[3, 3] = 1.0
        return transformation_matrix

    @staticmethod
    def get_ego_pose_dict(engo_info_dict):
        ego_translation = [engo_info_dict["egoX"], engo_info_dict["egoY"], engo_info_dict["egoZ"]]
        ego_rotation = [engo_info_dict["ego_quat0"], engo_info_dict["ego_quat1"],
                        engo_info_dict["ego_quat2"], engo_info_dict["ego_quat3"], ]
        ego_dict_ = {}
        ego_dict_.update({"rotation": ego_rotation})
        ego_dict_.update({"translation": ego_translation})
        return ego_dict_

    @staticmethod
    def invert_matrix_egopose_numpy(egopose):
        """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
        inverse_matrix = np.zeros((4, 4), dtype=np.float32)
        rotation = egopose[:3, :3]
        translation = egopose[:3, 3]
        inverse_matrix[:3, :3] = rotation.T
        inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
        inverse_matrix[3, 3] = 1.0
        return inverse_matrix

    @staticmethod
    def mat2pose_vec(matrix: torch.Tensor):
        """
        Converts a 4x4 pose matrix into a 6-dof pose vector
        Args:
            matrix (ndarray): 4x4 pose matrix
        Returns:
            vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
            rotation components (rx, ry, rz)
        """

        # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
        rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

        # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
        cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
        roty = torch.atan2(matrix[..., 0, 2], cosy)

        # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
        rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

        rotation = torch.stack((rotx, roty, rotz), dim=-1)

        # Extract translation params
        translation = matrix[..., :3, 3]
        return torch.cat((translation, rotation), dim=-1)

class Collater(object):
    def __init__(self,
                 target_width,
                 target_height,
                 embedding_dim):
        self.target_width = target_width
        self.target_height = target_height
        self.embedding_dim = embedding_dim
        self.position_encoding = SinePositionalEncoding(num_feats=self.embedding_dim)

    def __call__(self, batch):
        image_data = np.stack([item["image"] for item in batch]) # images
        image_data = torch.from_numpy(image_data)
        img_shape_list = np.stack([item["src_image_shape"] for item in batch])  # cls
        meta_data = ([item["meta_data"] for item in batch])  # cls
        meta_vector_label_info = ([item["vector_label_info"] for item in batch])  # cls

        #---------------------------------------------------#
        # ego motion data
        #---------------------------------------------------#
        meta_ego_motion_info = torch.stack([ torch.stack(item["ego_motion_info"], dim=0) for item in batch], dim=0)

        #---------------------------------------------------#
        #  vector label
        #---------------------------------------------------#
        gt_binary = np.stack([item["gt_binary"] for item in batch]) # seg
        gt_binary = torch.from_numpy(gt_binary)

        gt_offsetmap = np.stack([item["gt_offsetmap"] for item in batch]) # seg
        gt_offsetmap = torch.from_numpy(gt_offsetmap)

        gt_instancemap = np.stack([item["gt_instancemap"] for item in batch]) # seg
        gt_instancemap = torch.from_numpy(gt_instancemap)

        gt_classmap = np.stack([item["gt_classmap"] for item in batch]) # seg
        gt_classmap = torch.from_numpy(gt_classmap)

        gt_point_direction = np.stack([item["gt_point_direction"] for item in batch]) # seg
        gt_point_direction = torch.from_numpy(gt_point_direction)


        #---------------------------------------------------#
        #  Detector Label
        #---------------------------------------------------#
        gt_det_boxes = [item["gt_det_box"] for item in batch] # det boxes
        gt_det_labels = [item["gt_det_label"] for item in batch] # det boxes

        #---------------------------------------------------#
        #  Input
        #---------------------------------------------------#
        output_dict = dict()
        output_dict["image"] = image_data
        output_dict["src_image_shape"] = img_shape_list
        output_dict["img_metas"] = meta_data
        output_dict["vector_metas"] = meta_vector_label_info
        output_dict["ego_motion"] = meta_ego_motion_info

        # vector label
        output_dict["gt_binary"]=gt_binary
        output_dict["gt_offsetmap"]=gt_offsetmap
        output_dict["gt_instancemap"]=gt_instancemap
        output_dict["gt_classmap"]=gt_classmap
        output_dict["gt_point_direction"]=gt_point_direction

        # detection label
        output_dict["gt_det_bboxes"]=gt_det_boxes
        output_dict["gt_det_labels"]=gt_det_labels

        # pm tensor
        pm_matrix = np.stack([item["pm_matrix"] for item in batch])
        pm_matrix = torch.from_numpy(pm_matrix)
        output_dict["pm_matrix"]=pm_matrix

        # image folder name
        output_dict["img_name_list"]=[item["img_name_list"] for item in batch] # det boxes

        return output_dict

if __name__ == '__main__':
    #---------------------------------------------------#
    #  Parameter Settings
    #---------------------------------------------------#
    CFG_PATH = "../cfgs/ultrabev_stn_seq1_pretrain.yml"
    MODE = "train"
    BATCH_SIZE = 2
    NUM_WORKER = 0
    SAVE_VECTOR_LABEL = True

    #---------------------------------------------------#
    #  Config Loading
    #---------------------------------------------------#
    cfgs = yaml.safe_load(open(CFG_PATH))
    bevdata = BEVData(cfgs=cfgs,mode=MODE)
    use_half_direct = cfgs["vector"]["use_half_direct"]

    #---------------------------------------------------#
    #  Dataloader Object
    #---------------------------------------------------#
    trainloader = torch.utils.data.dataloader.DataLoader(bevdata,
                                                         batch_size=BATCH_SIZE,
                                                         num_workers=NUM_WORKER,
                                                         shuffle=True,
                                                         drop_last=False,
                                                         pin_memory=True,
                                                         collate_fn=bevdata.collate_fn)

    one_data = iter(trainloader).__next__()

    for key,value in one_data.items():
        if not isinstance(value,list):
            if value is not None:
                print(key, value.shape)
            else:
                print(key,"None")
        else:
            print(key)
            for elem in value:
                print(elem)

    if not os.path.exists("../bev_vector/sample"):
        os.makedirs("../bev_vector/sample")

    if SAVE_VECTOR_LABEL:
        gt_binary = one_data["gt_binary"]
        np.save("../bev_vector/sample/gt_binary.npy", gt_binary)

        gt_offsetmap = one_data["gt_offsetmap"]
        np.save("../bev_vector/sample/gt_offsetmap.npy", gt_offsetmap)

        gt_instancemap = one_data["gt_instancemap"]
        np.save("../bev_vector/sample/gt_instancemap.npy", gt_instancemap)

        gt_classmap = one_data["gt_classmap"]
        np.save("../bev_vector/sample/gt_classmap.npy", gt_classmap)

        if use_half_direct:
            gt_point_direction = one_data["gt_point_direction"]
            np.save("../bev_vector/sample/gt_point_direction_half.npy", gt_point_direction)
        else:
            gt_point_direction = one_data["gt_point_direction"]
            np.save("../bev_vector/sample/gt_point_direction_full.npy", gt_point_direction)

