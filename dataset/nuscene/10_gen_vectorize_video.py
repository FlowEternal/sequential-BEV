import numpy as np
import os
import cv2

from dataset.nuscene.Config import *

import warnings
warnings.filterwarnings("ignore")

import yaml

DELTA_X = 50
DELTA_Y = 80
txt_org = (0 + DELTA_X, 0 + DELTA_Y)
font_scale = 3
thickness = 6
color = (0, 0, 255)

COLOR_VIS_LIST = [
    (0,0,0),
    (255,255,255), # dynamic
    (255,0,0),
    (0,255,0),
    (0,128,255),
    (0,128,128),
    (255,0,255),
    (128,0,255)
    ]

OUTPUT_VIDEO_SIZE = (1680,720)

if __name__ == '__main__':

    # 数据的根目录
    output_dir = os.path.join(NUSCENE_DATA, str(INDEX_START) + "_" + str(INDEX_END))
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # ---------------------------------------------------#
    # 写数据
    # ---------------------------------------------------#
    video_path = os.path.join(NUSCENE_DATA, str(INDEX_START) + "_" + str(INDEX_END) + "_final_label.avi")
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(video_path, codec, 30, OUTPUT_VIDEO_SIZE)

    # ---------------------------------------------------#
    #  尺寸计算
    # ---------------------------------------------------#
    nx = [(row[1] - row[0]) / row[2] for row in [XBOUND, YBOUND, ZBOUND]]
    pxPerM = (float(nx[0]) / (XBOUND[1] - XBOUND[0]), float(nx[1]) / (YBOUND[1] - YBOUND[0]))
    M1 = int((CAR_LENGTH / 2.0 + 0.1) * pxPerM[0])
    M2 = int((CAR_WIDTH / 2.0 + 0.1) * pxPerM[1])

    for tmp_index in range(INDEX_START, INDEX_END + 1):
        print("processing folder %s" % tmp_index )
        front_left = yaml.safe_load(open(os.path.join(output_dir,str(tmp_index),"cameras","CAM_FRONT_LEFT.yaml")) )["image_name"]
        front = yaml.safe_load(open(os.path.join(output_dir,str(tmp_index),"cameras","CAM_FRONT.yaml")) )["image_name"]
        front_right = yaml.safe_load(open(os.path.join(output_dir,str(tmp_index),"cameras","CAM_FRONT_RIGHT.yaml")) )["image_name"]
        back_left = yaml.safe_load(open(os.path.join(output_dir,str(tmp_index),"cameras","CAM_BACK_LEFT.yaml")) )["image_name"]
        back = yaml.safe_load(open(os.path.join(output_dir,str(tmp_index),"cameras","CAM_BACK.yaml")) )["image_name"]
        back_right = yaml.safe_load(open(os.path.join(output_dir,str(tmp_index),"cameras","CAM_BACK_RIGHT.yaml")) )["image_name"]

        # image loading
        img_root_dir = os.path.join(output_dir,str(tmp_index),"images")
        img_front_left = cv2.imread( os.path.join(img_root_dir,front_left))
        img_front = cv2.imread(os.path.join(img_root_dir,front))
        img_front_right = cv2.imread(os.path.join(img_root_dir,front_right))
        img_back_left = cv2.imread(os.path.join(img_root_dir,back_left))
        img_back = cv2.imread(os.path.join(img_root_dir,back))
        img_back_right = cv2.imread(os.path.join(img_root_dir,back_right))

        # bev label loading
        label_dir = os.path.join(output_dir,str(tmp_index),"vis_vector", "vector.png")
        bev_label = cv2.imread(label_dir,cv2.IMREAD_COLOR)
        height, width = img_front.shape[0:2]

        # drawer
        img_draw = np.zeros([2*height,3*width,3],dtype=np.uint8)

        # front left
        img_front_left = cv2.putText(img_front_left,"front left",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[0:height,0:width,:] = img_front_left

        # front
        img_front = cv2.putText(img_front,"front",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[0:height,width: 2*width,:] = img_front

        # front right
        img_front_right = cv2.putText(img_front_right,"front right",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[0:height,2*width: 3*width,:] = img_front_right

        # back left
        img_back_left = cv2.putText(img_back_left,"back left",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[height:2*height,0: width,:] = img_back_left

        # back
        img_back = cv2.putText(img_back,"back",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[height:2*height,width: 2*width,:] = img_back

        # back right
        img_back_right = cv2.putText(img_back_right,"back right",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[height:2*height,2*width: 3*width,:] = img_back_right


        # bev label process
        bev_height, bev_width = bev_label.shape[0:2]
        cv2.rectangle(bev_label, (int(bev_width/2) - M1, int(bev_height/2) - M2),
                      (int(bev_width/2) + M1, int(bev_height/2) + M2), (0, 255, 255), -1)

        # 旋转90度
        bev_label = cv2.flip(bev_label, 1)
        bev_label = cv2.transpose(bev_label)

        # 再次resize
        img_draw = cv2.resize(img_draw,( OUTPUT_VIDEO_SIZE[0]-bev_label.shape[1],bev_label.shape[0]))
        total_image = cv2.hconcat([img_draw,bev_label])
        total_image = cv2.resize(total_image,OUTPUT_VIDEO_SIZE)

        out.write(total_image)

