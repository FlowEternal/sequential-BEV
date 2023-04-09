"""
Function: VectorBEV Model Defination
Author: Zhan Dong Xu
Date: 2021/11/11
"""

from abc import ABC
import numpy as np

import torch
import torch.nn
import torch.nn.functional as F

# Backbone
from net.resnet import resnet18, resnet34

# Spatial Transformation Layer
from net.stn import SpatialTransformer

# Camera Fusion
from net.fusion import MultiCamFusionEncoderLayer, MultiCamFusionDeoderLayer
from dataset.dataloader import SinePositionalEncoding

# Sequential Fusion
from net.temporal_model import TemporalModel, TemporalModelIdentity
from net.geometry import cumulative_warp_features

# Vector Header
from bev_vector.vector_header import VectorHeader

# Detection Header
from bev_det.centernet import CenterNetHead

#---------------------------------------------------#
#  VectorBEV Defination
#---------------------------------------------------#
class VectorBEV(torch.nn.Module, ABC):
    def __init__(self,cfgs, onnx_export = False, mask_path = None):
        super(VectorBEV, self).__init__()

        # save for reference
        self.cfgs = cfgs
        self.onnx_export = onnx_export
        self.mask_path = mask_path

        # multiview features fusion strategies
        # True: features => transformer encoder => transformer decoder => fused bev features
        # False: features => spatial transformation layers => transformer encoder => fused bev features
        self.use_transformer_decoder = self.cfgs["backbone"]["use_transformer_decoder"]

        #---------------------------------------------------#
        #  General Parameter Settings
        #---------------------------------------------------#
        self.net_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.net_input_height = self.cfgs["dataloader"]["network_input_height"]
        self.cam_num = self.cfgs["dataloader"]["cam_num"]
        self.layer_id = self.cfgs["backbone"]["layer_id"]
        self.x_bound = self.cfgs["dataloader"]["x_bound"]
        self.y_bound = self.cfgs["dataloader"]["y_bound"]
        self.bev_feat_ratio = self.cfgs["backbone"]["bev_feat_ratio"]

        self.feat_height = int(self.net_input_height / pow(2, self.layer_id + 1))
        self.feat_width = int(self.net_input_width / pow(2, self.layer_id + 1))
        self.bev_width = int((self.x_bound[1] - self.x_bound[0]) / (self.x_bound[2]))
        self.bev_height = int((self.y_bound[1] - self.y_bound[0]) / (self.y_bound[2]))
        self.bev_feat_width = int(self.bev_width / self.bev_feat_ratio)
        self.bev_feat_height = int(self.bev_height / self.bev_feat_ratio)

        #---------------------------------------------------#
        #  Backbone Definations
        #---------------------------------------------------#
        self.res_type = self.cfgs["backbone"]["resnet_type"]
        self.layers_dim = self.cfgs["backbone"]["dim_array"]
        if self.res_type == 18:
            self.backbone = resnet18(pretrained=False, layers_dim=self.layers_dim)
        else:
            self.backbone = resnet34(pretrained=False, layers_dim=self.layers_dim)

        #---------------------------------------------------#
        #  Fusion Layers
        #---------------------------------------------------#

        # Transformer Encoder Fusion
        self.attn_num_layers = self.cfgs["backbone"]["attn_num_layers"]
        self.attn_num_heads = self.cfgs["backbone"]["attn_num_heads"]
        self.attn_dropout = self.cfgs["backbone"]["attn_dropout"]

        self.ffn_feedforward_channels = self.cfgs["backbone"]["ffn_feedforward_channels"]
        self.ffn_num_fcs = self.cfgs["backbone"]["ffn_num_fcs"]
        self.ffn_dropout = self.cfgs["backbone"]["ffn_dropout"]

        self.final_encode_dim = self.cfgs["backbone"]["channels"]

        if self.use_transformer_decoder:
            self.embed_dim = self.layers_dim[self.layer_id-1]
        else:
            self.embed_dim = self.layers_dim[self.layer_id-1] * self.cam_num

        self.fusion_dict = {"num_layers": self.attn_num_layers,
                  "transformerlayers": {"type": "BaseTransformerLayer",
                                        "attn_cfgs": [{"type": 'MultiheadAttention', "embed_dims": self.embed_dim,
                                                       "num_heads": self.attn_num_heads, "dropout": self.attn_dropout}],
                                        "ffn_cfgs": [dict(type='FFN',
                                                          embed_dims=self.embed_dim,
                                                          feedforward_channels=self.ffn_feedforward_channels,
                                                          num_fcs=self.ffn_num_fcs,
                                                          ffn_drop=self.ffn_dropout,
                                                          act_cfg=dict(type='ReLU', inplace=True))],
                                        "operation_order": ("self_attn", "norm", "ffn", "norm")}
                  }

        self.fusion_multi_cams = MultiCamFusionEncoderLayer(**self.fusion_dict)

        # further encoding layer
        self.encoding = torch.nn.Conv2d(self.embed_dim, self.final_encode_dim, (3, 3), stride=1, padding=1)

        #---------------------------------------------------#
        #  BEVMapping Layer Defination
        #---------------------------------------------------#
        self.bev_mapping_layer = None
        self.spatial_transformer = None
        if self.use_transformer_decoder:
            # TransformerDecoder Config
            self.transformer_decoder_dict = {"num_layers":self.attn_num_layers,
              "transformerlayers":{"type":"TransformerDeoderLayer",
                         "attn_cfgs":{"type":'MultiheadAttention',
                                       "embed_dims":self.embed_dim,
                                       "num_heads":self.attn_num_heads,
                                       "dropout":self.attn_dropout},
                         "feedforward_channels": self.ffn_feedforward_channels,
                         "ffn_dropout": self.ffn_dropout,

                         "ffn_cfgs": dict(type='FFN',
                                         embed_dims=self.embed_dim,
                                         feedforward_channels=self.ffn_feedforward_channels,
                                         num_fcs=self.ffn_num_fcs,
                                         ffn_drop=self.ffn_dropout,
                                         act_cfg=dict(type='ReLU', inplace=True)),
                         "operation_order":("self_attn","norm","cross_attn","norm","ffn","norm")}
              }

            self.bev_mapping_layer = MultiCamFusionDeoderLayer(**self.transformer_decoder_dict).cuda()

        else:
            # spatial transformer layer
            self.use_mask = self.cfgs["backbone"]["use_mask"]
            self.spatial_transformer = SpatialTransformer(feat_width= self.feat_width,
                                                          feat_height=self.feat_height,
                                                          x_bound=self.x_bound,
                                                          y_bound=self.y_bound,
                                                          bev_feat_ratio=self.bev_feat_ratio,
                                                          use_mask=self.use_mask,
                                                          deploy=self.onnx_export,
                                                          mask_path=self.mask_path,
                                                          )

        #---------------------------------------------------#
        #  TemporaryModuleDefination
        #---------------------------------------------------#
        self.seq_num = self.cfgs["dataloader"]["seq_num"]
        self.seq_module_name = self.cfgs["backbone"]["seq_module_name"]
        self.spatial_extent = (self.x_bound[1], self.y_bound[1])

        self.start_out_channels = self.cfgs["backbone"]["start_out_channels"]
        self.extra_in_channels = self.cfgs["backbone"]["extra_in_channels"]
        self.inbetween_layers = self.cfgs["backbone"]["inbetween_layers"]
        self.pyramid_pooling = self.cfgs["backbone"]["pyramid_pooling"]

        self.receptive_field = self.seq_num
        self.final_encode_dim_temporary = self.final_encode_dim
        if self.seq_num > 1:
            self.final_encode_dim_temporary +=6

        if self.seq_num == 1:
            self.temporal_model = TemporalModelIdentity(self.final_encode_dim_temporary, self.receptive_field)
        else:
            self.temporal_model = TemporalModel(
                self.final_encode_dim_temporary,
                self.receptive_field,
                input_shape=(self.bev_height, self.bev_width),
                start_out_channels=self.start_out_channels,
                extra_in_channels=self.extra_in_channels,
                n_spatial_layers_between_temporal_layers=self.inbetween_layers,
                use_pyramid_pooling=self.pyramid_pooling,
            )

        #---------------------------------------------------#
        #  1.Vector Header
        #---------------------------------------------------#
        self.vector_scale_ratio = self.cfgs["vector"]["vector_scale_ratio"]
        self.cluster_feat_dim = self.cfgs["vector"]["cluster_feat_dim"]
        self.exist_condidence_loss = self.cfgs["vector"]["exist_condidence_loss"]
        self.nonexist_confidence_loss = self.cfgs["vector"]["nonexist_confidence_loss"]
        self.sisc_loss = self.cfgs["vector"]["sisc_loss"]
        self.disc_loss = self.cfgs["vector"]["disc_loss"]
        self.delta_v =self.cfgs["vector"]["delta_v"]
        self.delta_d = self.cfgs["vector"]["delta_d"]
        self.thresh = self.cfgs["vector"]["thresh"]
        self.threshold_instance = self.cfgs["vector"]["threshold_instance"]
        self.class_num = self.cfgs["vector"]["class_num"]
        self.threshold_non_exist = self.cfgs["vector"]["threshold_non_exist"]
        self.sample_distance = self.cfgs["vector"]["sample_distance"]
        self.resolution = self.cfgs["vector"]["resolution"]
        self.threshold_remove = self.cfgs["vector"]["threshold_remove"]
        self.angle_list = self.cfgs["vector"]["angle_list"]
        self.cluster_min_num = self.cfgs["vector"]["cluster_min_num"]
        self.use_half_direct = self.cfgs["vector"]["use_half_direct"]
        self.use_resize_conv_os = self.cfgs["vector"]["use_resize_conv_os"]

        self.grid_x = self.bev_feat_width
        self.grid_y = self.bev_feat_height
        self.x_size = self.bev_width
        self.y_size = self.bev_height
        self.vector_header = VectorHeader(
                                     self.final_encode_dim,
                                     self.cluster_feat_dim,
                                     exist_condidence_loss=self.exist_condidence_loss,
                                     nonexist_confidence_loss=self.nonexist_confidence_loss,
                                     sisc_loss=self.sisc_loss,
                                     disc_loss=self.disc_loss,
                                     grid_x=self.grid_x,
                                     grid_y=self.grid_y,
                                     x_size=self.x_size,
                                     y_size=self.y_size,
                                     delta_v=self.delta_v,
                                     delta_d=self.delta_d,
                                     thresh=self.thresh,
                                     threshold_instance=self.threshold_instance,
                                     resize_ratio=self.bev_feat_ratio,
                                     class_num=self.class_num,
                                     threshold_non_exist=self.threshold_non_exist,
                                     sample_distance=self.sample_distance,
                                     resolution=self.resolution,
                                     threshold_remove=self.threshold_remove,
                                     angle_list=self.angle_list,
                                     up_sample_time=int(np.log2(self.bev_feat_ratio / self.vector_scale_ratio)),
                                     cluster_min_num=self.cluster_min_num,
                                     use_half_direct=self.use_half_direct,
                                     use_resize_conv_os=self.use_resize_conv_os,
        )

        self.vector_center_heatmap = self.cfgs["vector"]["vector_center_heatmap"]
        self.vector_offset = self.cfgs["vector"]["vector_offset"]
        self.vector_instance = self.cfgs["vector"]["vector_instance"]
        self.vector_direct = self.cfgs["vector"]["vector_direct"]
        self.vector_classify = self.cfgs["vector"]["vector_classify"]

        #---------------------------------------------------#
        #  2.Detection Header
        #---------------------------------------------------#
        self.class_list = cfgs["detection"]["class_list"][1:]
        self.num_classes = cfgs["detection"]["num_classes"]
        assert len(self.class_list) == self.num_classes
        self.in_channel = cfgs["detection"]["in_channel"]
        self.feat_channel = cfgs["detection"]["feat_channel"]

        self.loss_center_heatmap = cfgs["detection"]["loss_center_heatmap"]
        self.loss_wh = cfgs["detection"]["loss_wh"]
        self.loss_offset = cfgs["detection"]["loss_offset"]
        self.loss_rotation = cfgs["detection"]["loss_rotation"]
        self.test_cfg = cfgs["detection"]["test_cfg"]

        # Detection Header Defination
        self.center_head = CenterNetHead(
            in_channel=self.in_channel,
            feat_channel=self.feat_channel,
            num_classes=self.num_classes,
            loss_center_heatmap=self.loss_center_heatmap,
            loss_wh=self.loss_wh,
            loss_offset=self.loss_offset,
            loss_rotation=self.loss_rotation,
            train_cfg=None,
            test_cfg=self.test_cfg,
            init_cfg=None
        )

        # Loss Function For Detection
        self.loss_det = self.center_head.loss

        #---------------------------------------------------#
        #  Mask Related Constant Tensor
        #---------------------------------------------------#
        if self.use_transformer_decoder:
            # Preparation
            self.position_encoding = SinePositionalEncoding(
                num_feats= int(self.layers_dim[self.layer_id - 1] * 0.5))

            # Used For TransformerEncoder Mask
            self.mask_input = np.zeros([1, self.feat_height, self.feat_width * self.cam_num], dtype=np.bool)

            # Used For TransformerEncoder Pose Embedding
            self.bev_pos_embed = self.position_encoding(torch.from_numpy(self.mask_input)).float().numpy()

            # Used For TransformerDecoder Query Embedding
            bev_pos_tensor = torch.from_numpy(np.zeros([1, self.bev_feat_height, self.bev_feat_width], dtype=np.bool))
            self.transformer_bev_pos_embed = self.position_encoding(bev_pos_tensor).float().numpy()


        else:
            # Preparation
            self.position_encoding = SinePositionalEncoding(num_feats=self.layers_dim[self.layer_id-1] * int(self.cam_num/2) )

            # Used For TransformerEncoder Mask
            self.mask_input = np.zeros([1, self.bev_feat_height, self.bev_feat_width], dtype=np.bool)

            # Used For TransformerEncoder Pose Embedding
            self.bev_pos_embed = self.position_encoding(torch.from_numpy(self.mask_input)).float().numpy()

            # Used For TransformerDecoder Query Embedding
            bev_pos_tensor = torch.from_numpy(np.zeros([1, self.bev_feat_height, self.bev_feat_width], dtype=np.bool))
            self.transformer_bev_pos_embed = self.position_encoding(bev_pos_tensor).float().numpy()

    def forward(self, x, pm_matrix, ego_motion_matrix = None,mode = "train"):
        if mode =="train":
            bev_pos_embed = torch.from_numpy(self.bev_pos_embed).unsqueeze(1).repeat(x.shape[0], self.seq_num, 1, 1, 1).cuda()
            bev_mask_input = torch.from_numpy(self.mask_input).unsqueeze(1).repeat(x.shape[0], self.seq_num, 1, 1, 1).cuda()
            query_embed = torch.from_numpy(self.transformer_bev_pos_embed).unsqueeze(1).repeat(x.shape[0], self.seq_num, 1,1, 1).cuda()

        else:
            bev_pos_embed = torch.from_numpy(self.bev_pos_embed).unsqueeze(1).repeat(1, self.seq_num, 1, 1, 1).cuda()
            bev_mask_input = torch.from_numpy(self.mask_input).unsqueeze(1).repeat(1, self.seq_num, 1, 1, 1).cuda()
            query_embed = torch.from_numpy(self.transformer_bev_pos_embed).unsqueeze(1).repeat(1, self.seq_num, 1,1, 1).cuda()

        return self.forward_(x, pm_matrix, ego_motion_matrix, bev_pos_embed, bev_mask_input, query_embed, mode)

    def forward_(self, x, pm_matrix, ego_motion_matrix, bev_pos_embed, bev_mask_input, query_embed=None, mode = "train"):
        # Feature Extraction
        batch_size = x.shape[0]
        x = x.view(batch_size * self.seq_num * self.cam_num, 3, self.net_input_height, self.net_input_width)
        pm_matrix = pm_matrix.view(batch_size * self.seq_num * self.cam_num, 3, 3)
        feat = self.backbone(x, True)[self.layer_id]

        # Fusion
        if self.use_transformer_decoder:
            bev_feat = feat.view(batch_size * self.seq_num , self.cam_num, self.embed_dim, -1).permute(0,2,1,3).contiguous().view(batch_size * self.seq_num, self.embed_dim, -1).permute(2,0,1).contiguous()
            bev_pos_embed = bev_pos_embed.view(batch_size * self.seq_num, self.embed_dim, -1).permute(2, 0, 1).contiguous()
            bev_mask_input = bev_mask_input.view(batch_size * self.seq_num, -1).bool()
            bev_memory = self.fusion_multi_cams(
                query=bev_feat,
                key=None,
                value=None,
                query_pos=bev_pos_embed,
                query_key_padding_mask=bev_mask_input)

            query_embed = query_embed.view(batch_size * self.seq_num,  self.embed_dim, -1).permute(2, 0, 1)
            target = torch.zeros_like(query_embed)
            encoding_tmp = self.bev_mapping_layer(
                query=target,
                key=bev_memory,
                value=bev_memory,
                key_pos=bev_pos_embed,
                query_pos=query_embed,
                key_padding_mask=bev_mask_input)

            encoding_tmp = encoding_tmp[-1].view( self.bev_feat_height,  self.bev_feat_width,  batch_size * self.seq_num,  self.embed_dim).permute(2, 3, 0, 1)
            bev_encoding = self.encoding(encoding_tmp)

        else:
            bev_feat = self.spatial_transformer(feat, pm_matrix)
            bev_feat = bev_feat.contiguous().view(batch_size * self.seq_num, self.embed_dim, -1).permute(2, 0, 1).contiguous()
            bev_pos_embed = bev_pos_embed.view(batch_size * self.seq_num, self.embed_dim, -1).permute(2, 0, 1).contiguous()
            bev_mask_input = bev_mask_input.view(batch_size * self.seq_num, -1).bool()
            bev_memory = self.fusion_multi_cams(
                query=bev_feat,
                key=None,
                value=None,
                query_pos=bev_pos_embed,
                query_key_padding_mask=bev_mask_input)
            bev_memory = bev_memory.view(self.bev_feat_height, self.bev_feat_width, batch_size * self.seq_num, self.embed_dim).permute(2, 3, 0, 1).contiguous()
            bev_encoding = self.encoding(bev_memory)

        #---------------------------------------------------#
        #  Temporary Module Fusion
        #---------------------------------------------------#
        bev_encoding = bev_encoding.view(batch_size, self.seq_num, *bev_encoding.shape[1:])

        # Warp past features to the present's reference frame
        bev_encoding = cumulative_warp_features(bev_encoding.clone(), ego_motion_matrix,mode='bilinear',
                                                spatial_extent=self.spatial_extent,export=self.onnx_export)

        b, s, c = ego_motion_matrix.shape
        h, w = bev_encoding.shape[-2:]
        future_egomotions_spatial = ego_motion_matrix.view(b, s, c, 1, 1).expand(b, s, c, h, w)
        # at time 0, no egomotion so feed zero vector
        if self.seq_num > 1:
            future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                                   future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
            bev_encoding = torch.cat([bev_encoding, future_egomotions_spatial], dim=-3)
        bev_encoding = self.temporal_model(bev_encoding)[:, -1]

        #---------------------------------------------------#
        #  bev vector headear
        #---------------------------------------------------#
        out_confidence, out_offset, out_instance, out_direct, out_cls = self.vector_header(bev_encoding)

        #---------------------------------------------------#
        #  bev center headear
        #---------------------------------------------------#
        outs_detect = self.center_head([bev_encoding])

        if mode == "train":
            #---------------------------------------------------#
            #  Deccode Vector Header
            #---------------------------------------------------#
            output_dict = {}
            output_dict.update({"confidence":out_confidence})
            output_dict.update({"offset":out_offset})
            output_dict.update({"instance":out_instance})
            output_dict.update({"direct":out_direct})
            output_dict.update({"cls":out_cls})

            #---------------------------------------------------#
            #  Decode Detection Header
            #---------------------------------------------------#
            output_dict.update({"detection":outs_detect})
            return output_dict

        else:
            # post process detect
            center_heatmap_pred, wh_pred, offset_pred, rotation_pred = outs_detect
            hmax = F.max_pool2d(center_heatmap_pred[0], 3, stride=1, padding=1)
            heat_point = (hmax == center_heatmap_pred[0]).float() * center_heatmap_pred[0]
            topk_scores, topk_inds = torch.topk(heat_point.view(1, -1), 100)  # 这里取top 100
            return out_confidence, out_offset, out_instance, out_direct, out_cls, \
                   topk_scores, topk_inds, wh_pred[0], offset_pred[0], rotation_pred[0]

    # Key Function -- Calculate Loss
    def cal_loss(self, pred_dict, gt_dict):
        loss_dict = {}

        # vector
        gt_binary = gt_dict["gt_binary"]
        gt_offsetmap = gt_dict["gt_offsetmap"]
        gt_instancemap = gt_dict["gt_instancemap"]
        gt_classmap = gt_dict["gt_classmap"]
        gt_point_direction = gt_dict["gt_point_direction"]

        out_confidence = pred_dict["confidence"]
        out_offset = pred_dict["offset"]
        out_instance = pred_dict["instance"]
        out_direct = pred_dict["direct"]
        out_cls = pred_dict["cls"]
        outs = [out_confidence, out_offset, out_instance, out_direct, out_cls]
        center_map_loss, offset_loss, cluster_loss, cls_loss, direct_loss = \
            self.vector_header.cal_loss_vector(gt_binary, gt_offsetmap, gt_instancemap, gt_point_direction, gt_classmap, outs)


        loss_dict.update({"loss_center_map":center_map_loss * self.vector_center_heatmap})
        loss_dict.update({"loss_offset":offset_loss * self.vector_offset})
        loss_dict.update({"loss_cluster":cluster_loss * self.vector_instance})
        loss_dict.update({"loss_cls":cls_loss * self.vector_direct})
        loss_dict.update({"loss_direct":direct_loss * self.vector_classify})


        # detection
        target_shape = (self.bev_height, self.bev_width)
        gt_bboxes = gt_dict["gt_det_bboxes"]
        gt_labels = gt_dict["gt_det_labels"]
        loss_cal_inputy = pred_dict["detection"] + (gt_bboxes, gt_labels, target_shape)
        loss_dict_detection = self.loss_det(*loss_cal_inputy)
        loss_dict["loss_detect_center_heatmap"] = loss_dict_detection["loss_center_heatmap"]
        loss_dict["loss_detect_wh"] = loss_dict_detection["loss_wh"]
        loss_dict["loss_detect_offset"] = loss_dict_detection["loss_offset"]
        loss_dict["loss_detect_rotation"] = loss_dict_detection["loss_rotation"]

        return loss_dict


if __name__ == '__main__':
    import warnings, yaml, time
    warnings.filterwarnings("ignore")

    #---------------------------------------------------#
    #  Parameter Settings
    #---------------------------------------------------#
    # CFG_PATH = "cfgs/ultrabev_stn_seq3_finetune.yml"
    CFG_PATH = "cfgs/ultrabev_transformer_seq1_pretrain.yml"

    batch_size = 1

    #---------------------------------------------------#
    #  Parameters Loading
    #---------------------------------------------------#
    cfgs = yaml.safe_load(open(CFG_PATH))
    vectorbev = VectorBEV(cfgs=cfgs).cuda()
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    cam_num = cfgs["dataloader"]["cam_num"]
    seq_num = cfgs["dataloader"]["seq_num"]

    layer_id = cfgs["backbone"]["layer_id"]
    layers_dim = cfgs["backbone"]["dim_array"]

    # for transfomer encoder - decoder
    use_transformer_decoder = cfgs["backbone"]["use_transformer_decoder"]
    feat_height = int(net_input_height / pow(2, layer_id + 1))
    feat_width = int(net_input_width / pow(2, layer_id + 1))

    x_bound = cfgs["dataloader"]["x_bound"]
    y_bound = cfgs["dataloader"]["y_bound"]
    feat_ratio = cfgs["backbone"]["bev_feat_ratio"]
    bev_width = int((x_bound[1] - x_bound[0]) / (x_bound[2]))
    bev_height = int((y_bound[1] - y_bound[0]) / (y_bound[2]))
    bev_feat_width = int(bev_width / feat_ratio)
    bev_feat_height = int(bev_height / feat_ratio)

    #---------------------------------------------------#
    #  Input Definations
    #---------------------------------------------------#
    dummy_input = torch.randn((batch_size, seq_num, cam_num, 3, net_input_height, net_input_width)).cuda()
    dummy_pm_matrix = torch.randn(batch_size, seq_num, cam_num, 3, 3).cuda()
    dummy_ego_motion_matrix = torch.randn(batch_size, seq_num, 6).cuda()

    #---------------------------------------------------#
    #  Inference Test
    #---------------------------------------------------#
    avg_runtime = 0.0
    for _ in range(50):
        tic = time.time()
        ouptut = vectorbev(dummy_input, dummy_pm_matrix, dummy_ego_motion_matrix)
        torch.cuda.synchronize()

        print("inference time: %i" %(1000*(time.time() - tic)))
        avg_runtime += 1000*(time.time() - tic)
    print("average time: %i" % (avg_runtime/50))

    #---------------------------------------------------#
    #  Deploy Test
    #---------------------------------------------------#
    dummy_input_deploy = torch.randn((1, seq_num, cam_num, 3, net_input_height, net_input_width)).cuda()
    dummy_pm_matrix_deploy = torch.randn(1, seq_num, cam_num, 3, 3).cuda()
    dummy_ego_motion_matrix_deploy = torch.randn(1, seq_num, 6).cuda()

    ouptut = vectorbev(dummy_input_deploy, dummy_pm_matrix_deploy, dummy_ego_motion_matrix_deploy, "deploy")
    for out_ in ouptut:
        print(out_.shape)