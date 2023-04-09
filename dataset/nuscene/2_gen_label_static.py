import os
import torch
import numpy as np
import cv2

# import matplotlib as mpl
# mpl.use('Agg')
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion


import warnings
warnings.filterwarnings("ignore")

# from dataset.nuscene.Config import *
from Config import *


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage",
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

def plot_nusc_map_(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,(XBOUND[1],YBOUND[1]), poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)

def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx, vis_dir,label_dir):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]
    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot_angel_plane = np.arctan2(rot[1, 0], rot[0, 0])

    # patch
    cx = egopose['translation'][0]
    cy = egopose['translation'][1]
    patch_box = (cx, cy, XBOUND[1] - XBOUND[0], YBOUND[1] - YBOUND[0])

    # Default orientation where North is up
    patch_angle = 90.0 + rot_angel_plane / 3.1415 * 180

    # mask提取
    canvas_size = ( int((XBOUND[1] - XBOUND[0])/XBOUND[2]), int((YBOUND[1] - YBOUND[0])/YBOUND[2]))
    map_mask = nusc_maps[map_name].get_map_mask(patch_box, patch_angle, STATIC_LAYER_NAME, canvas_size)
    map_mask = map_mask[:,:,::-1]

    # 保存结果
    for idx, tmp_name in enumerate(STATIC_LAYER_NAME):
        tmp_img = map_mask[idx]
        tmp_img = cv2.flip(tmp_img, 1)
        tmp_img = cv2.transpose(tmp_img)
        tmp_img = cv2.flip(tmp_img, 1)
        tmp_img = cv2.transpose(tmp_img)
        np.save( os.path.join(label_dir,"bev_static_"+tmp_name+".npy"),tmp_img)

    if INCREASE_SPACE:
        visual_mask = np.zeros_like(map_mask[0])
        for idx, id_color in enumerate(STATIC_LAYER_ID):
            visual_mask[map_mask[idx] == 1] = id_color

        visual_mask = cv2.flip(visual_mask, 1)
        visual_mask = cv2.transpose(visual_mask)
        visual_mask = cv2.flip(visual_mask, 1)
        visual_mask = cv2.transpose(visual_mask)
        cv2.imwrite(os.path.join(vis_dir,"bev_static_visual.png"),visual_mask)

    if DEBUG:
        poly_names = [STATIC_LAYER_NAME[0], STATIC_LAYER_NAME[3]]
        line_names = [STATIC_LAYER_NAME[1], STATIC_LAYER_NAME[2]]
        plt.figure(figsize=( int(nx[1]/100), int(nx[0]/100)))
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot_angel_plane), np.sin(rot_angel_plane)])
        lmap = get_local_map(nusc_maps[map_name], center,(XBOUND[1],YBOUND[1]), poly_names, line_names)
        for name in poly_names:
            for la in lmap[name]:
                pts = (la - bx) / dx
                plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
        for la in lmap['road_divider']:
            pts = (la - bx) / dx
            plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
        for la in lmap['lane_divider']:
            pts = (la - bx) / dx
            plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)

        plt.xlim((nx[1],0))
        plt.ylim((0,nx[0]))
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')  # 去掉坐标轴
        plt.savefig(os.path.join(vis_dir,"bev_static_matplot.png"),bbox_inches='tight', pad_inches=0)

def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch[0],
        center[1] - stretch[1],
        center[0] + stretch[0],
        center[1] + stretch[1],
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,layer_names=layer_names,mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def gen_static_label(input_rec, save_dir):
    vis_dir = os.path.join(save_dir, "visual")
    if INCREASE_SPACE:
        if not os.path.exists(vis_dir): os.makedirs(vis_dir)

    label_dir = os.path.join(save_dir,"gt_label")
    if not os.path.exists(label_dir): os.makedirs(label_dir)

    plot_nusc_map(input_rec, nusc_maps, nusc, scene2map, dx, bx, vis_dir,label_dir)

if __name__ == '__main__':

    DEBUG = False
    #---------------------------------------------------#
    #  导出参数
    #---------------------------------------------------#
    # 数据的根目录
    output_dir = os.path.join(NUSCENE_DATA,str(INDEX_START) + "_" + str(INDEX_END))
    if not os.path.exists(output_dir):os.makedirs(output_dir)

    # nuscene数据对象
    nusc = NuScenes(version='v1.0-{}'.format(VERSION),dataroot=os.path.join(NUSCENE_DATA, VERSION),verbose=True)
    samples = nusc.sample

    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    # nuscene地图对象
    nusc_maps = get_nusc_maps(os.path.join(NUSCENE_DATA,VERSION))

    # 场景和map联系
    scene2map = {}
    for rec in nusc.scene:
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    # 尺寸计算
    dx, bx, nx = gen_dx_bx(XBOUND, YBOUND, ZBOUND)
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    #---------------------------------------------------#
    #  离线提取数据
    #---------------------------------------------------#
    for tmp_index in range(INDEX_START, INDEX_END+1):
        rec = samples[tmp_index]
        print("processing record %s %i" %(rec["token"], tmp_index) )

        output_dir_tmp = os.path.join(output_dir, str(tmp_index))
        if not os.path.exists(output_dir_tmp):os.makedirs(output_dir_tmp)

        gen_static_label(rec, output_dir_tmp)