import os
import torch
import numpy as np
import cv2
from pyquaternion import Quaternion

# 读入nuscene数据集
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from dataset.nuscene.Config import *

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def gen_dynamic_label(rec, vis_dir):
    egopose = nusc.get('ego_pose',nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    trans = -np.array(egopose['translation'])
    rot = Quaternion(egopose['rotation']).inverse

    gt_label_dir = os.path.join(vis_dir, "gt_label")
    if not os.path.exists(gt_label_dir): os.makedirs(gt_label_dir)

    gt_dynamic_txt = os.path.join(gt_label_dir, "object.txt")
    gt_label_writer = open(gt_dynamic_txt, "w")


    for class_name in DYNAMIC_OBJECT_LIST:
        name_id_dict = list()
        img_instance = np.zeros((nx[0], nx[1]))

        counter = 0
        for idx,tok in enumerate(rec['anns']):
            inst = nusc.get('sample_annotation', tok)

            # 如果不是在这个category里面 直接跳过
            category_name = inst['category_name'].split('.')[0]
            if len(inst['category_name'].split('.')) > 1:
                specific_name = inst['category_name'].split('.')[1]
            else:
                specific_name = "none"
            if not category_name == class_name:continue

            # 如果不可见 直接跳过
            if int(inst['visibility_token']) == 1:continue
            # index + 1
            counter+=1

            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)
            pts = box.bottom_corners()[:2].T
            pts = np.round((pts - bx[:2] + dx[:2]/2.) / dx[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]

            # draw bounding box
            # color_list = [100, 128 ,192, 255]
            # for index in range(pts.shape[0]):
            #     pt1 = (pts[index][0], pts[index][1])
            #     pt2 = (pts[(index+1)%4][0], pts[(index+1)%4][1])
            #     cv2.line(img_instance, pt1, pt2, color_list[index], thickness=2)

            # id - name list
            name_id_dict.append((counter,specific_name))

            cv2.fillPoly(img_instance, [pts], counter)

            # label写入检测文件
            pts = pts[::-1]

            det_info = '{},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                class_name,pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1])


            gt_label_writer.writelines(det_info)

        # segmentation
        seg_name = "bev_dynamic_" + class_name + "_seg"
        np.save( os.path.join(gt_label_dir,seg_name +".npy"),img_instance)

        if DEBUG:
            plt.imshow(img_instance)
            plt.xticks([])  # 去掉x轴
            plt.yticks([])  # 去掉y轴
            plt.axis('off')  # 去掉坐标轴
            plt.savefig(os.path.join(vis_dir,"visual",seg_name +".png"),bbox_inches='tight', pad_inches=0)

        # name id list
        name_id_list_name = "bev_dynamic_" + class_name + "_id_list"
        np.save(os.path.join(gt_label_dir,name_id_list_name+".npy"), np.array(name_id_dict))



if __name__ == '__main__':
    DEBUG = False
    #---------------------------------------------------#
    #  导出参数
    #---------------------------------------------------#
    # 数据的根目录
    output_dir = os.path.join(NUSCENE_DATA,str(INDEX_START) + "_" + str(INDEX_END))
    if not os.path.exists(output_dir):os.makedirs(output_dir)

    # nuscene
    nusc = NuScenes(version='v1.0-{}'.format(VERSION),dataroot=os.path.join(NUSCENE_DATA, VERSION),verbose=True)
    samples = nusc.sample

    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    dx, bx, nx = gen_dx_bx(XBOUND, YBOUND, ZBOUND)
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    #---------------------------------------------------#
    #  离线提取数据
    #---------------------------------------------------#
    for tmp_index in range(INDEX_START, INDEX_END+1):
        rec = samples[tmp_index]
        print("processing record %s, %i" %(rec["token"], tmp_index) )

        output_dir_tmp = os.path.join(output_dir, str(tmp_index))
        if not os.path.exists(output_dir_tmp):os.makedirs(output_dir_tmp)

        gen_dynamic_label(rec, output_dir_tmp)