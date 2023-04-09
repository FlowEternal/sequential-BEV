"""
Function: Gen Test Images From TrainVal Set
Author: Zhan Dong Xu
Date: 2021/11/11
"""

import shutil
import os

def get_correspond_img(img_list, cam_name):
    for tmp_img_ in img_list:
        if cam_name == "CAM_FRONT":
            if "CAM_FRONT" in tmp_img_ and "CAM_FRONT_LEFT" not in tmp_img_ and "CAM_FRONT_RIGHT" not in tmp_img_:
                return tmp_img_
        elif cam_name == "CAM_BACK":
            if "CAM_BACK" in tmp_img_ and "CAM_BACK_LEFT" not in tmp_img_ and "CAM_BACK_RIGHT" not in tmp_img_:
                return tmp_img_
        else:
            if cam_name in tmp_img_:
                return tmp_img_


if __name__ == '__main__':
    root_dir = "/data/zdx/Data/data_nuscene"
    num_to_get = 100
    start_index = 20000
    target_dir = os.path.join(root_dir, "demo_video")
    cam_list = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
    for cam_name in cam_list:
        if not os.path.exists(os.path.join(target_dir, cam_name)):
            os.makedirs(os.path.join(target_dir, cam_name))

    data_dir = os.path.join(root_dir, "0_34148")
    list_folder = os.listdir(data_dir)[start_index: start_index + num_to_get]

    for cam_name in cam_list:
        target_tmp_dir = os.path.join(target_dir, cam_name)

        for tmp_folder in list_folder:
            src_tmp_dir = os.path.join(data_dir,tmp_folder,"images")
            src_tmp_img_list = os.listdir(src_tmp_dir)

            target_img_name = get_correspond_img(src_tmp_img_list, cam_name)

            src_tmp_img_path = os.path.join(src_tmp_dir, target_img_name)
            dst_tmp_img_path = os.path.join(target_tmp_dir, target_img_name)
            shutil.copy(src_tmp_img_path, dst_tmp_img_path)