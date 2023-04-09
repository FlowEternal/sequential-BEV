# ==================================================================
# Author    : Dongxu Zhan
# Time      : 2021/7/14 22:12
# File      : 11_gen_ego_motion.py
# Function  : generate ego motion at every timestamp
# ==================================================================

import os
import numpy as np
import shutil
import yaml
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# from dataset.nuscene.Config import *
from Config import *

def gen_sample(nusc, rec, output_dir):
    output_dir_meta = os.path.join(output_dir, "ego")
    if not os.path.exists(output_dir_meta): os.makedirs(output_dir_meta)

    # ---------------------------------------------------#
    #  保存ego vehicle和lidar的外参信息
    # ---------------------------------------------------#
    sample_data_token = rec['data']['LIDAR_TOP']
    egopose = nusc.get(
        'ego_pose', nusc.get('sample_data', sample_data_token)['ego_pose_token']
    )

    # 保存ego vehicle的信息
    calib_ego = {}
    trans_ego = np.array(egopose['translation'])
    calib_ego["egoX"] = float(trans_ego[0])
    calib_ego["egoY"] = float(trans_ego[1])
    calib_ego["egoZ"] = float(trans_ego[2])

    rot_ego = np.array(egopose['rotation'])
    calib_ego["ego_quat0"] = float(rot_ego[0])
    calib_ego["ego_quat1"] = float(rot_ego[1])
    calib_ego["ego_quat2"] = float(rot_ego[2])
    calib_ego["ego_quat3"] = float(rot_ego[3])
    print("ego vehicle to world translation:")
    print(trans_ego)
    print("ego vehicle to world rotation:")
    print(rot_ego)
    with open(os.path.join(output_dir_meta, "ego.yaml"), "w", encoding="utf-8") as f:yaml.dump(calib_ego, f)

def process_one_sample(tmp_index):
    rec = samples[tmp_index]
    print("processing record %s %i" % (rec["token"],tmp_index))

    # output dir
    output_dir_tmp = os.path.join(output_dir, str(tmp_index))
    if not os.path.exists(output_dir_tmp): os.makedirs(output_dir_tmp)

    # generate sample
    gen_sample(nusc, rec, output_dir_tmp)

if __name__ == '__main__':
    #---------------------------------------------------#
    #  导出参数
    #---------------------------------------------------#
    # 数据的根目录
    output_dir = os.path.join(NUSCENE_DATA,str(INDEX_START) + "_" + str(INDEX_END))
    if not os.path.exists(output_dir):os.makedirs(output_dir)

    # nuscene对象
    nusc = NuScenes(version='v1.0-{}'.format(VERSION),dataroot=os.path.join(NUSCENE_DATA, VERSION),verbose=True)
    samples = nusc.sample
    print(f"total sample number is {len(samples)}")

    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    #---------------------------------------------------#
    #  离线提取数据
    #---------------------------------------------------#
    for tmp_index in range(INDEX_START, INDEX_END+1):
        process_one_sample(tmp_index)