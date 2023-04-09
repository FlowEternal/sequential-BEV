"""
Function: Multicamera Fusion Module
Author: Zhan Dong Xu
Date: 2021/11/11
"""

from abc import ABC

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer

#---------------------------------------------------#
#  Transfomer Decoder Defination
#---------------------------------------------------#
@TRANSFORMER_LAYER.register_module()
class TransformerDeoderLayer(BaseTransformerLayer, ABC):
    """TransformerDeoderLayer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=None,
                 norm_cfg=None,
                 ffn_num_fcs=2,
                 **kwargs):

        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        if act_cfg is None:
            act_cfg = dict(type='ReLU', inplace=True)
        super(TransformerDeoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)

        assert len(operation_order) == 6
        assert set(operation_order) == {'self_attn', 'norm', 'cross_attn', 'ffn'}


#---------------------------------------------------#
#  Transformer Encoder Defination
#---------------------------------------------------#
class MultiCamFusionEncoderLayer(TransformerLayerSequence, ABC):
    """MultiCamFusionEncoderLayer.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=None, **kwargs):
        super(MultiCamFusionEncoderLayer, self).__init__(*args, **kwargs)
        if post_norm_cfg is None:
            post_norm_cfg = dict(type='LN')
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(MultiCamFusionEncoderLayer, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


#---------------------------------------------------#
#  Transfomer Decoder Defination
#---------------------------------------------------#
class MultiCamFusionDeoderLayer(TransformerLayerSequence, ABC):
    def __init__(self,
                 *args,
                 post_norm_cfg=None,
                 return_intermediate=False,
                 **kwargs):

        super(MultiCamFusionDeoderLayer, self).__init__(*args, **kwargs)
        if post_norm_cfg is None:
            post_norm_cfg = dict(type='LN')
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


# ---------------------------------------------------#
#  Test Script
# ---------------------------------------------------#
if __name__ == '__main__':
    import warnings, yaml, time
    warnings.filterwarnings("ignore")

    # ---------------------------------------------------#
    #  Parameter Setting
    # ---------------------------------------------------#
    CFG_PATH = "../cfgs/ultrabev_stn_seq3_pretrain.yml"
    batch_size = 1

    # load config
    cfgs = yaml.safe_load(open(CFG_PATH))
    use_transformer_decoder = cfgs["backbone"]["use_transformer_decoder"]

    # calculate size
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    cam_num = cfgs["dataloader"]["cam_num"]
    x_bound = cfgs["dataloader"]["x_bound"]
    y_bound = cfgs["dataloader"]["y_bound"]

    # calculate model parameters
    layer_id = cfgs["backbone"]["layer_id"]
    layers_dim = cfgs["backbone"]["dim_array"]
    bev_feat_ratio = cfgs["backbone"]["bev_feat_ratio"]

    feat_height = int(net_input_height / pow(2, layer_id + 1))
    feat_width = int(net_input_width / pow(2, layer_id + 1))
    bev_width = int((x_bound[1] - x_bound[0]) / (x_bound[2]))
    bev_height = int((y_bound[1] - y_bound[0]) / (y_bound[2]))
    bev_feat_width = int(bev_width / bev_feat_ratio)
    bev_feat_height = int(bev_height / bev_feat_ratio)

    if use_transformer_decoder:
        embed_dim = layers_dim[layer_id-1]
    else:
        embed_dim = layers_dim[layer_id-1] * cam_num

    final_encode_dim = cfgs["backbone"]["channels"]

    # ---------------------------------------------------#
    #  TransformerEncoder Definations
    # ---------------------------------------------------#
    attn_num_layers =  cfgs["backbone"]["attn_num_layers"]
    attn_num_heads =  cfgs["backbone"]["attn_num_heads"]
    attn_dropout =  cfgs["backbone"]["attn_dropout"]
    ffn_feedforward_channels =  cfgs["backbone"]["ffn_feedforward_channels"]
    ffn_num_fcs =  cfgs["backbone"]["ffn_num_fcs"]
    ffn_dropout =  cfgs["backbone"]["ffn_dropout"]

    kwargs_fusion = {"num_layers":attn_num_layers,
              "transformerlayers":{"type":"BaseTransformerLayer",
                         "attn_cfgs":[{"type":'MultiheadAttention',
                                       "embed_dims":embed_dim,
                                       "num_heads":attn_num_heads,
                                       "dropout":attn_dropout}],

                         "ffn_cfgs":[dict(type = 'FFN',
                                          embed_dims = embed_dim,
                                          feedforward_channels = ffn_feedforward_channels,
                                          num_fcs = ffn_num_fcs,
                                          ffn_drop = ffn_dropout,
                                          act_cfg = dict(type='ReLU', inplace=True))],
                         "operation_order":("self_attn","norm","ffn","norm")}
              }

    memory_layer = MultiCamFusionEncoderLayer(**kwargs_fusion).cuda()


    # ---------------------------------------------------#
    #  TransformerDecoder Definations
    # ---------------------------------------------------#
    kwargs_bev = {"num_layers":attn_num_layers,
              "transformerlayers":{"type":"TransformerDeoderLayer",
                         "attn_cfgs":{"type":'MultiheadAttention',
                                       "embed_dims":embed_dim,
                                       "num_heads":attn_num_heads,
                                       "dropout":attn_dropout},
                         "feedforward_channels": ffn_feedforward_channels,
                         "ffn_dropout": ffn_dropout,

                         "ffn_cfgs": dict(type='FFN',
                                         embed_dims=embed_dim,
                                         feedforward_channels=ffn_feedforward_channels,
                                         num_fcs=ffn_num_fcs,
                                         ffn_drop=ffn_dropout,
                                         act_cfg=dict(type='ReLU', inplace=True)),
                         "operation_order":("self_attn","norm","cross_attn","norm","ffn","norm")}
              }

    bev_mapping_layer = MultiCamFusionDeoderLayer(**kwargs_bev).cuda()

    #---------------------------------------------------#
    #  Input Defination
    #---------------------------------------------------#
    query_embed = torch.randn(batch_size, embed_dim, bev_feat_height, bev_feat_width).cuda()
    query_embed = query_embed.view(batch_size, embed_dim, -1).permute(2, 0, 1)
    target = torch.zeros_like(query_embed)

    if use_transformer_decoder:
        dummy_input = torch.randn(batch_size * cam_num, embed_dim, feat_height, feat_width).cuda()
        bev_pos_embed = torch.randn(batch_size , embed_dim, feat_height, feat_width * cam_num).cuda()
        bev_mask_input = torch.zeros(batch_size, feat_height, feat_width * cam_num).bool().cuda()

        # change shape
        dummy_input = dummy_input.view(batch_size, cam_num, embed_dim, -1).permute(0,2,1,3).contiguous().view(batch_size, embed_dim, -1).permute(2,0,1).contiguous()
        bev_pos_embed = bev_pos_embed.view(batch_size, embed_dim, -1).permute(2, 0, 1)
        bev_mask_input = bev_mask_input.view(batch_size, -1)

    else:
        dummy_input = torch.randn(batch_size, embed_dim, bev_feat_height, bev_feat_width).cuda()
        bev_pos_embed = torch.randn(batch_size , embed_dim, bev_feat_height, bev_feat_width).cuda()
        bev_mask_input = torch.zeros(batch_size, bev_feat_height, bev_feat_width).bool().cuda()

        # change shape
        dummy_input = dummy_input.view(batch_size, embed_dim, -1).permute(2, 0, 1)
        bev_pos_embed = bev_pos_embed.view(batch_size, embed_dim, -1).permute(2, 0, 1)
        bev_mask_input = bev_mask_input.view(batch_size, -1)

    #---------------------------------------------------#
    #  Cameras Fusion
    #---------------------------------------------------#
    # inference speed
    for iter_ in range(10):
        tic = time.time()
        torch.cuda.synchronize()
        if use_transformer_decoder:
            memory = memory_layer(
                query=dummy_input,
                key=None,
                value=None,
                query_pos=bev_pos_embed,
                query_key_padding_mask=bev_mask_input)

            encoding = bev_mapping_layer(
                query=target,
                key=memory,
                value=memory,
                key_pos=bev_pos_embed,
                query_pos=query_embed,
                key_padding_mask=bev_mask_input)

            encoding = encoding[-1].view(bev_feat_height, bev_feat_width, batch_size, embed_dim).permute(2, 3, 0, 1)
            encoding = torch.nn.Conv2d(embed_dim, final_encode_dim, (3, 3), stride=1, padding=1).cuda()(encoding)

        else:
            memory = memory_layer(
                query=dummy_input,
                key=None,
                value=None,
                query_pos=bev_pos_embed,
                query_key_padding_mask=bev_mask_input)
            memory = memory.view(bev_feat_height, bev_feat_width, batch_size, embed_dim).permute(2, 3, 0, 1)
            encoding = torch.nn.Conv2d(embed_dim, final_encode_dim, (3, 3), stride=1, padding=1).cuda()(memory)
        torch.cuda.synchronize()

        if iter_ == 0:
            print("Fused Feature Dimension:")
            print(encoding.shape)

        print("inference time is %f" %(1000*(time.time() - tic)))

