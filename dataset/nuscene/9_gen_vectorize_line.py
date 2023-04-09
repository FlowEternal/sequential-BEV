import os
import numpy as np

from nuscenes import NuScenes
from bev_vector.devkit.data.vector_map import VectorizedLocalMap
from bev_vector.devkit.data.rasterize import rasterize_map

import warnings
warnings.filterwarnings("ignore")

# from dataset.nuscene.Config import *
from Config import *

if __name__ == '__main__':
    #---------------------------------------------------#
    #  导出参数
    #---------------------------------------------------#
    # 数据的根目录
    output_dir = os.path.join(NUSCENE_DATA,str(INDEX_START) + "_" + str(INDEX_END))
    if not os.path.exists(output_dir):os.makedirs(output_dir)

    # 尺寸计算
    patch_h = YBOUND[1] - YBOUND[0]
    patch_w = XBOUND[1] - XBOUND[0]
    canvas_h = int(patch_h / YBOUND[2])
    canvas_w = int(patch_w / XBOUND[2])
    patch_size = (patch_h, patch_w)
    canvas_size = (canvas_h, canvas_w)

    # nuscene数据对象
    nusc = NuScenes(version='v1.0-{}'.format(VERSION),dataroot=os.path.join(NUSCENE_DATA, VERSION),verbose=True)
    samples = nusc.sample

    # vector map class
    vector_map = VectorizedLocalMap(os.path.join(NUSCENE_DATA, VERSION),
                                    patch_size=patch_size,
                                    canvas_size=canvas_size,
                                    sample_dist=SAMPLE_DISTANCE,
                                    patch_margin=PATCH_MARGIN)

    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    #---------------------------------------------------#
    #  离线矢量化线
    #---------------------------------------------------#
    for tmp_index in range(INDEX_START, INDEX_END+1):

        # tmp_index =15
        rec = samples[tmp_index]
        print("processing record %s %i" %(rec["token"], tmp_index) )

        vis_save_dir = os.path.join(output_dir, str(tmp_index), "vis_vector")
        if not os.path.exists(vis_save_dir): os.makedirs(vis_save_dir)

        # 开始进行矢量化
        location = nusc.get('log', nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        pred_map, confidence_level,line_lists = rasterize_map(vectors, patch_size, canvas_size,
                                                              MAXCHANNEL,
                                                              THICKNESS,
                                                              KERNEL_SIZE_DILATION_CROSSWALK,
                                                              min_length_crosswalk=MIN_CROSSWALK_LENGTH,
                                                              min_length_lane=MIN_LINE_LENGTH,
                                                              min_road_edge=MIN_ROAD_EDGE,
                                                              concat_edge_threshold=CONCAT_EDGE_THRESHOLD,
                                                              concat_edge_threshold_crosswalk=CONCAT_EDGE_THRESHOLD_CROSSWALK,
                                                              random_color=RANDOM_COLOR,
                                                              debug=DEBUG,expand_num=EXPAND_PTS_NUM,
                                                              increase_space=INCREASE_SPACE_VECTOR,
                                                              vis_save_dir=vis_save_dir)

        # 保存结果
        save_dir = os.path.join(output_dir, str(tmp_index),"gt_label","vector_label.npy")
        np.save(save_dir, line_lists)

