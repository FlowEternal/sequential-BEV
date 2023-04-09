# ==================================================================
# Author    : Dongxu Zhan
# Time      : 2021/7/14 22:12
# File      : 1_gen_raw_sample.py
# Function  : generate nuscene original image and camera parameter
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
    output_dir_img = os.path.join(output_dir, "images")
    output_dir_cam = os.path.join(output_dir, "cameras")
    output_dir_lidar = os.path.join(output_dir, "lidar")
    output_dir_meta = os.path.join(output_dir, "ego")

    if not os.path.exists(output_dir_img): os.makedirs(output_dir_img)
    if not os.path.exists(output_dir_cam): os.makedirs(output_dir_cam)

    #---------------------------------------------------#
    #  信息显示
    #---------------------------------------------------#
    if DEBUG:
        # render point cloud in camera
        for cam in CAMERA:
            nusc.render_pointcloud_in_image(rec['token'], camera_channel=cam)

        # render point cloud
        nusc.render_sample_data(rec['data']['LIDAR_TOP'], nsweeps=1, underlay_map=False)
        print(nusc.get('scene', rec['scene_token'])["name"])

    if INCREASE_SPACE:
        if not os.path.exists(output_dir_lidar): os.makedirs(output_dir_lidar)
        if not os.path.exists(output_dir_meta): os.makedirs(output_dir_meta)

        # ---------------------------------------------------#
        #  保存ego vehicle和lidar的外参信息
        # ---------------------------------------------------#
        sample_data_token = rec['data']['LIDAR_TOP']
        sample_info = nusc.get('sample_data', sample_data_token)
        lidar_src_path = os.path.join(nusc.dataroot, sample_info['filename'])

        lidar_dst_path = os.path.join(output_dir_lidar, os.path.basename(sample_info["filename"]))
        shutil.copy(lidar_src_path, lidar_dst_path)

        # 保存lidar的信息
        calib_lidar = {}
        lidar_info = nusc.get('calibrated_sensor', sample_info['calibrated_sensor_token'])
        trans_lidar = np.array(lidar_info['translation'])
        calib_lidar["lidarX"] = float(trans_lidar[0])
        calib_lidar["lidarY"] = float(trans_lidar[1])
        calib_lidar["lidarZ"] = float(trans_lidar[2])

        rot_lidar = np.array(lidar_info['rotation'])
        calib_lidar["lidar_quat0"] = float(rot_lidar[0])
        calib_lidar["lidar_quat1"] = float(rot_lidar[1])
        calib_lidar["lidar_quat2"] = float(rot_lidar[2])
        calib_lidar["lidar_quat3"] = float(rot_lidar[3])

        if DEBUG:
            print("lidar to ego vehicle translation x y z:")
            print(trans_lidar)
            print("lidar to ego vehicle rotation matrix:")
            print(rot_lidar)

        with open(os.path.join(output_dir_lidar, "lidar.yaml"), "w", encoding="utf-8") as f:yaml.dump(calib_lidar, f)

        # 保存ego vehicle的信息
        calib_ego = {}
        egopose = nusc.get('ego_pose',sample_info['ego_pose_token'])
        trans_ego = np.array(egopose['translation'])
        calib_ego["egoX"] = float(trans_ego[0])
        calib_ego["egoY"] = float(trans_ego[1])
        calib_ego["egoZ"] = float(trans_ego[2])

        rot_ego = np.array(egopose['rotation'])
        calib_ego["ego_quat0"] = float(rot_ego[0])
        calib_ego["ego_quat1"] = float(rot_ego[1])
        calib_ego["ego_quat2"] = float(rot_ego[2])
        calib_ego["ego_quat3"] = float(rot_ego[3])

        if DEBUG:
            print("ego vehicle to world translation:")
            print(trans_ego)
            print("ego vehicle to world rotation:")
            print(rot_ego)

        with open(os.path.join(output_dir_meta, "ego.yaml"), "w", encoding="utf-8") as f:yaml.dump(calib_ego, f)

    #---------------------------------------------------#
    #  保存camera 内外参信息
    #---------------------------------------------------#
    calib_cam = {}
    for cam in CAMERA:
        samp = nusc.get('sample_data', rec['data'][cam])
        imgname = os.path.join(nusc.dataroot, samp['filename'])
        output_img_path = os.path.join(output_dir_img,os.path.basename(samp['filename']))
        shutil.copy(imgname,output_img_path)

        # 显示信息
        if DEBUG:
            nusc.render_sample_data(samp['token'])

        # 相机内参保存
        sens = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
        intrinsic_matrix = sens['camera_intrinsic']

        if DEBUG:
            print(cam)
            print("INTRINSIC PARAMETERS")
            print(intrinsic_matrix)

        calib_cam["fx"] = float(intrinsic_matrix[0][0])
        calib_cam["fy"] = float(intrinsic_matrix[1][1])
        calib_cam["px"] = float(intrinsic_matrix[0][2])
        calib_cam["py"] = float(intrinsic_matrix[1][2])

        # 相机外参保存
        trans_cam = np.array(sens['translation'])
        calib_cam["camX"] = float(trans_cam[0])
        calib_cam["camY"] = float(trans_cam[1])
        calib_cam["camZ"] = float(trans_cam[2])

        rot_cam = np.array(sens['rotation'])
        calib_cam["cam_quat0"] = float(rot_cam[0])
        calib_cam["cam_quat1"] = float(rot_cam[1])
        calib_cam["cam_quat2"] = float(rot_cam[2])
        calib_cam["cam_quat3"] = float(rot_cam[3])

        # 对应图片名称保存
        calib_cam["image_name"] = os.path.basename(samp['filename'])

        # 保存yaml
        output_dir_cam_this = os.path.join(output_dir_cam,cam+".yaml")
        with open(output_dir_cam_this, "w", encoding="utf-8") as f:
            yaml.dump(calib_cam, f)

        if DEBUG:
            # 相机外参打印
            print("Camera to ego vehicle extrinsic")
            print("--- rotation yaw pitch roll ---")
            print(Quaternion(sens['rotation']).yaw_pitch_roll)
            print("--- rotation matrix ---")
            print(Quaternion(sens['rotation']).rotation_matrix)
            print("--- rotation quaternions (w ri rj rk) ---")
            print(rot_cam)
            print("--- translation tx ty tz ---")
            print(sens['translation'])
            print("")

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