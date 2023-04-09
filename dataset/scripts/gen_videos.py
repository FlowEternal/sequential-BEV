"""
Function: Generate Video From Image Folder
Author: Zhan Dong Xu
Date: 2021/11/11
"""

import glob
import cv2
import os

def batch_img_to_video(output_path, im_list):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), target_fps, target_size)

    for src_path in im_list:
        org_img = cv2.imread(src_path)
        out.write(org_img)
        print("finishing reading image %s" %src_path)

    out.release()

if __name__ == "__main__":
    root_dir = "/data/zdx/Data/data_nuscene/demo_video"
    CAMERA = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]

    # root_dir = "D:\\Data\\BEV\\own\\pack3"
    # CAMERA = ["2021_07_08_14_40_19_053_cam0","2021_07_08_14_40_19_053_cam1","2021_07_08_14_40_19_053_cam2","2021_07_08_14_40_19_053_cam3","2021_07_08_14_40_19_053_cam4","2021_07_08_14_40_19_053_cam5"]

    # root_dir = "D:\\Data\\MULTITASK_TEST\\Desktop"
    # CAMERA = ["cam0","cam1","cam2"]


    MAX_IMG_NUM = 5000
    # IMG_ORG_SIZE = (1920,1080)
    IMG_ORG_SIZE = (1600,900)

    FPS = 20
    SUFFIX = "*.jpg"

    counter = 0
    for tmp_cam in CAMERA:
        # input_ paramters
        im_path = os.path.join(root_dir,tmp_cam)
        output_path = os.path.join(root_dir,tmp_cam) + ".avi"
        target_size = IMG_ORG_SIZE
        target_fps = FPS

        # images
        src_img_list = sorted(glob.glob(os.path.join(im_path,SUFFIX)))
        batch_img_to_video(output_path,src_img_list)
        counter+=1
        print(tmp_cam + "" + str(counter))
