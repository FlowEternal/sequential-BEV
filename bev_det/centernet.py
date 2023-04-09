# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC

import cv2
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet.models import build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target

from bev_det.gaussian_target import (get_local_maximum, get_topk_from_heatmap,transpose_and_gather_feat)
from bev_det.base_dense_head import BaseDenseHead
from bev_det.dense_test_mixins import BBoxTestMixin
from bev_det.rot_det_metric import eval_mAP

def _build_head(in_channel, feat_channel, out_channel):
    """Build head for each branch."""
    layer = nn.Sequential(
        nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(feat_channel, out_channel, kernel_size=1))
    return layer


class CenterNetHead(BaseDenseHead, BBoxTestMixin, ABC):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input_ feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=None,
                 loss_wh=None,
                 loss_offset=None,
                 loss_rotation=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CenterNetHead, self).__init__(init_cfg)
        if loss_offset is None:
            loss_offset = dict(type='L1Loss', loss_weight=1.0)
        if loss_wh is None:
            loss_wh = dict(type='L1Loss', loss_weight=0.1)
        if loss_center_heatmap is None:
            loss_center_heatmap = dict(
                type='GaussianFocalLoss', loss_weight=1.0)
        if loss_rotation is None:
            loss_rotation = dict(
                type='L1Loss', loss_weight=1.0)

        self.num_classes = num_classes
        self.heatmap_head = _build_head(in_channel, feat_channel,num_classes)
        self.wh_head = _build_head(in_channel, feat_channel, 2)
        self.offset_head = _build_head(in_channel, feat_channel, 2)

        # added by zdx rotation angel prediction
        self.rotation_head = _build_head(in_channel, feat_channel, 1)

        # self.loss_center_heatmap = build_loss(loss_center_heatmap)
        # self.loss_wh = build_loss(loss_wh)
        # self.loss_offset = build_loss(loss_offset)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_rotation = build_loss(loss_rotation)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head, self.rotation_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        rotation_pred = self.rotation_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred, rotation_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             rotation_preds,
             gt_bboxes,
             gt_labels,
             target_shape,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            rotation_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            target_shape (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        rotation_prd = rotation_preds[0]
        target_result, avg_factor = self.get_targets_with_rotation(gt_bboxes, gt_labels,center_heatmap_pred.shape,target_shape)

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # 取出rotation相关标签
        rotation_target = target_result['rotation_target']
        rotation_offset_target_weight = target_result['rotation_target_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh( wh_pred,wh_target,wh_offset_target_weight,avg_factor=avg_factor * 2)
        loss_rotation = self.loss_rotation( rotation_prd,rotation_target,rotation_offset_target_weight,avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(offset_pred,offset_target,wh_offset_target_weight,avg_factor=avg_factor * 2)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_rotation=loss_rotation)

    def get_targets_with_rotation(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        rotation_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w])
        rotation_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 1, feat_h, feat_w])

        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]

            # gt box (x1,y1,x2,y2,x3,y3,x4,y4)
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]] + gt_bbox[:, [4]] + gt_bbox[:, [6]]) * width_ratio / 4
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]] + gt_bbox[:, [5]] + gt_bbox[:, [7]]) * height_ratio / 4
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                # scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                # scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                gt_bbox_resize = gt_bbox[j].view(4,2)
                pt1, pt2, pt3, pt4 = arrange_point(gt_bbox_resize)  # 这里必须要顺时针顺次的点
                scale_box_w = torch.sqrt((pt2[1] - pt1[1]) * (pt2[1] - pt1[1]) + (pt2[0] - pt1[0]) * (pt2[0] - pt1[0])) * width_ratio
                scale_box_h = torch.sqrt((pt4[1] - pt1[1]) * (pt4[1] - pt1[1]) + (pt4[0] - pt1[0]) * (pt4[0] - pt1[0])) * height_ratio
                theta = torch.atan( -(pt2[1] - pt1[1])/ (pt2[0] - pt1[0] + 1e-8) )

                radius = gaussian_radius([scale_box_h, scale_box_w],min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]

                gen_gaussian_target(center_heatmap_target[batch_id, ind],[ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                # rotation
                rotation_target[batch_id, 0, cty_int, ctx_int] = theta

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

                rotation_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight,
            rotation_target=rotation_target,
            rotation_target_weight=rotation_target_weight
            )
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   rotation_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            rotation_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == len(rotation_preds) == 1
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        border_pixs = [img_meta['border'] for img_meta in img_metas]

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            rotation_preds[0],
            img_metas[0]['batch_input_shape'],
            k=self.test_cfg["topk"],
            kernel=self.test_cfg["local_maximum_kernel"])

        batch_border = batch_det_bboxes.new_tensor(
            border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        batch_det_bboxes[..., :4] -= batch_border

        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       rotation_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            rotation_pred (Tensor): rotation predict, shape (B, 1, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        rotation = transpose_and_gather_feat(rotation_pred, batch_index)

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y,rotation[..., 0]], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4].contiguous(),
                                       bboxes[:, -1].contiguous(), labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def arrange_point(pts_tmp):

    # 对点进行对齐操作
    tmp_array = np.array(pts_tmp.detach().cpu().numpy())
    x_list = tmp_array[:, 0]
    y_list = tmp_array[:, 1]
    min_x_id = np.argmin(x_list)

    pt_tmp_x = tmp_array[min_x_id][0]

    counter = 0
    id_list = []
    for idx in range(len(x_list)):
        if pt_tmp_x == x_list[idx]:
            counter += 1
            id_list.append(idx)

    id_first = min_x_id
    if len(id_list) == 2:
        tmp_y1 = y_list[id_list[0]]
        tmp_y2 = y_list[id_list[1]]
        if tmp_y1 > tmp_y2:
            id_first = id_list[1]
        else:
            id_first = id_list[0]

    pt1 = pts_tmp[id_first]
    pt2 = pts_tmp[(id_first + 1) % 4]
    pt3 = pts_tmp[(id_first + 2) % 4]
    pt4 = pts_tmp[(id_first + 3) % 4]
    return pt1,pt2,pt3,pt4

if __name__ == '__main__':
    import numpy as np
    # import cv2
    # pt1, pt2, pt3, pt4 = arrange_point([[40,50],[50,40],[20,10],[10,20]])
    # pt1, pt2, pt3, pt4 = arrange_point([[20,50],[40,50],[40,10],[20,10]]) # 这里必须要顺时针顺次的点
    # pt1, pt2, pt3, pt4 = arrange_point([[40,50],[20,50],[20,10],[40,10]]) # 逆时针是不行的
    # rectangle = cv2.minAreaRect(np.array([pt1,pt2,pt3,pt4]))
    # center = rectangle[0]
    # width = np.sqrt( (pt2[1]-pt1[1]) *(pt2[1]-pt1[1]) + (pt2[0]-pt1[0]) *(pt2[0]-pt1[0]) )
    # height = np.sqrt( (pt4[1]-pt1[1]) *(pt4[1]-pt1[1]) + (pt4[0]-pt1[0]) *(pt4[0]-pt1[0]) )
    # width_, height_ = rectangle[1]
    # theta = np.arctan2((pt2[1]-pt1[1]),(pt2[0]-pt1[0] + 1e-8)) * 180 / 3.1415926

    import yaml
    CFG_PATH = "../cfgs/ultrabev_stn_seq3.yml"
    cfgs = yaml.safe_load(open(CFG_PATH))

    # input
    target_shape = (256, 512)
    channels = cfgs["backbone"]["channels"]
    x_bound = cfgs["dataloader"]["x_bound"]
    y_bound = cfgs["dataloader"]["y_bound"]
    feat_ratio = cfgs["backbone"]["bev_feat_ratio"]
    bev_width = int((x_bound[1] - x_bound[0]) / (x_bound[2]))
    bev_height = int((y_bound[1] - y_bound[0]) / (y_bound[2]))
    bev_feat_width = int(bev_width / feat_ratio)
    bev_feat_height = int(bev_height / feat_ratio)
    dummy_input = [torch.randn((2, channels, bev_feat_height, bev_feat_width)).cuda()]

    # paramter for detect head
    class_list = cfgs["detection"]["class_list"][1:]
    in_channel = cfgs["detection"]["in_channel"]
    feat_channel = cfgs["detection"]["feat_channel"]
    num_classes = cfgs["detection"]["num_classes"]
    loss_center_heatmap = cfgs["detection"]["loss_center_heatmap"]
    loss_wh = cfgs["detection"]["loss_wh"]
    loss_offset = cfgs["detection"]["loss_offset"]
    loss_rotation = cfgs["detection"]["loss_rotation"]
    test_cfg = cfgs["detection"]["test_cfg"]

    center_head = CenterNetHead(
                in_channel = in_channel,
                feat_channel = feat_channel,
                num_classes = num_classes,
                loss_center_heatmap = loss_center_heatmap,
                loss_wh = loss_wh,
                loss_offset = loss_offset,
                loss_rotation=loss_rotation,
                train_cfg = None,
                test_cfg = test_cfg,
                init_cfg = None
    ).cuda()

    # =========================================
    # inference test
    # =========================================
    center_heatmap_pred, wh_pred, offset_pred, rotation_pred = center_head(dummy_input)
    print(center_heatmap_pred[0].shape)
    print(wh_pred[0].shape)
    print(offset_pred[0].shape)
    print(rotation_pred[0].shape)

    outs = (center_heatmap_pred, wh_pred, offset_pred, rotation_pred)

    # =========================================
    # training
    # =========================================
    gt_bboxes = []
    gt_bbox_one = torch.tensor([[20,30,30,20,20,10,10,20],[20,30,30,20,20,10,10,20]],dtype=torch.float32).cuda()
    gt_bbox_two = torch.tensor([[20,30,30,20,20,10,10,20],[20,30,30,20,20,10,10,20]],dtype=torch.float32).cuda()
    gt_bboxes.append(gt_bbox_one)
    gt_bboxes.append(gt_bbox_two)

    gt_labels = []
    gt_label_one = torch.tensor([0,0],dtype=torch.int).cuda()
    gt_label_two = torch.tensor([0,0],dtype=torch.int).cuda()
    gt_labels.append(gt_label_one)
    gt_labels.append(gt_label_two)

    dummy_loss_input = (center_heatmap_pred, wh_pred, offset_pred, rotation_pred, gt_bboxes, gt_labels, target_shape)
    loss_dict = center_head.loss(*dummy_loss_input)
    for key, value in loss_dict.items():
        print(key, value)

    # =========================================
    # demo
    # =========================================
    import numpy as np
    img_meta = dict()
    img_meta["filename"] = "demo/demo_test.jpg"
    img_meta["ori_shape"] = (256, 512, 3)
    img_meta["img_shape"] = (256, 512, 3)
    img_meta["pad_shape"] = (256, 512, 3)
    img_meta["scale_factor"] = [1., 1., 1., 1.]
    img_meta["flip"] = False
    img_meta["flip_direction"] = None
    img_meta["img_norm_cfg"] = {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                                'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                                'to_rgb': True}

    img_meta["border"] = [0. ,256., 0., 512.]
    img_meta["batch_input_shape"] = (256, 512)
    img_metas = [img_meta]
    results_list = center_head.get_bboxes(*outs, img_metas, rescale=True)
    bbox_results = [
        bbox2result(det_bboxes, det_labels, num_classes)
        for det_bboxes, det_labels in results_list
    ]

    # =========================================
    # validation
    # =========================================
    import os
    img_name_list = ["img1.jpg", "img2.jpg"]

    # gt 准备
    gt_labels_dir = "../bev_det/eval_gt"
    if not os.path.exists(gt_labels_dir): os.makedirs(gt_labels_dir)
    for img_name in img_name_list:
        gt_path = os.path.join(gt_labels_dir, img_name.replace(".jpg", ".txt"))
        gt_writer = open(gt_path, "w")
        for i in range(10):
            gt_writer.writelines("vehicle_light,238,29,238,28,238,28,238,29\n")
        gt_writer.close()

    # predict 准备
    pred_labels_dir = "../bev_det/eval_pred"
    if not os.path.exists(pred_labels_dir): os.makedirs(pred_labels_dir)
    for idx in range(len(bbox_results)):
        tmp_img_path = img_name_list[idx]
        target_pred_txt_path = os.path.join(pred_labels_dir,tmp_img_path.replace(".jpg",".txt"))
        writer_this = open(target_pred_txt_path, "w")

        bbox_result = bbox_results[idx]
        for cls_id, one_class_result in enumerate(bbox_result):
            if one_class_result.shape[0] == 0:
                continue
            else:
                this_cls_num = one_class_result.shape[0]
                for i in range(this_cls_num):
                    x1 = one_class_result[i,0]
                    y1 = one_class_result[i,1]
                    x2 = one_class_result[i,2]
                    y2 = one_class_result[i,3]
                    rot_angel = one_class_result[i,4]
                    score = one_class_result[i,5]

                    x = (x1 + x2) /2.0
                    y = (y1 + y2) /2.0
                    w = max((x2 - x1),0)
                    h = max((y2 - y1),0)
                    pts = cv2.boxPoints(((x,y),(w,h),rot_angel)).reshape(-1)

                    det_info = '{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                        class_list[cls_id],
                        score,
                        pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7])

                    writer_this.writelines(det_info)

        writer_this.close()

    ROOT_DIR = "../bev_det"

    mAP = eval_mAP(ROOT_DIR=ROOT_DIR,
                   VALID_DETECT_GROUND_TRUTH=gt_labels_dir,
                   VALID_DETECT_RESULT=pred_labels_dir,
                   use_07_metric=False)
    print("mAP is: %.2f" %mAP)
