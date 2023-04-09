import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import cv2

import matplotlib.pyplot as plt

from dataset.nuscene.Config import *

import shutil

import warnings
warnings.filterwarnings("ignore")

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

if __name__ == '__main__':
    DEBUG = False
    # ---------------------------------------------------#
    #  尺寸计算
    # ---------------------------------------------------#
    dx, bx, nx = gen_dx_bx(XBOUND, YBOUND, ZBOUND)

    # 数据的根目录
    output_dir = os.path.join(NUSCENE_DATA, str(INDEX_START) + "_" + str(INDEX_END))
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for tmp_index in range(INDEX_START, INDEX_END + 1):
        print("processing folder %s" % tmp_index)
        otuput_dir_vis = os.path.join(output_dir,str(tmp_index),"visual")
        output_dir_tmp = os.path.join(output_dir, str(tmp_index),"gt_label")
        label_dict = dict()
        for dynamic_name in DYNAMIC_OBJECT_LIST:
            label_dict[dynamic_name] = {}

        #---------------------------------------------------#
        #  动态信息提取
        #---------------------------------------------------#
        for dynamic_name in DYNAMIC_OBJECT_LIST:
            instance_label_path = os.path.join(output_dir_tmp,"bev_dynamic_"+ dynamic_name+"_seg.npy")
            id_list_label_path = os.path.join(output_dir_tmp, "bev_dynamic_" + dynamic_name + "_id_list.npy")
            label_dict[dynamic_name].update({"instance":np.load(instance_label_path)})
            label_dict[dynamic_name].update({"id_list":np.load(id_list_label_path)})

        #---------------------------------------------------#
        #  静态信息提取
        #---------------------------------------------------#
        for static_name in STATIC_LAYER_NAME:
            static_label_path = os.path.join(output_dir_tmp,"bev_static_"+ static_name+".npy")
            label_dict[static_name] = np.load(static_label_path)

            if USING_LINE:
                if static_name in REGION2LINE:
                    # 进行区域边缘线提取
                    tmp_mask = label_dict[static_name]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,KERNEL_SIZE[static_name])
                    erodImg = cv2.erode(tmp_mask, kernel)
                    dilateImg = cv2.dilate(tmp_mask, kernel)
                    edgeImg = dilateImg - erodImg

                    # 保存结果
                    label_dict[static_name] = edgeImg.astype(np.uint8)

                    # 可视化
                    if INCREASE_SPACE:
                        tmp_save_path_refine = os.path.join(otuput_dir_vis, "refine")
                        if not os.path.exists(tmp_save_path_refine): os.makedirs(tmp_save_path_refine)

                        tmp_save_path = os.path.join(tmp_save_path_refine,"bev_static_"+static_name+"_edge.png")
                        cv2.imwrite(tmp_save_path,edgeImg.astype(np.uint8)*255)

        if USING_LINE:
            # ---------------------------------------------------#
            #  精细化操作 --- 这个非常重要
            # ---------------------------------------------------#

            # 如果人行道和路沿都在 精细化人行道
            if "ped_crossing" in STATIC_LAYER_NAME and "drivable_area" in STATIC_LAYER_NAME:
                ped_edge = label_dict["ped_crossing"]
                road_edge = label_dict["drivable_area"]
                intersect = ped_edge * road_edge

                # 可视化
                if INCREASE_SPACE:
                    tmp_save_path_refine = os.path.join(otuput_dir_vis,"refine")
                    if not os.path.exists(tmp_save_path_refine): os.makedirs(tmp_save_path_refine)

                    tmp_save_path = os.path.join(tmp_save_path_refine, "ped_road_intersect.png")
                    cv2.imwrite(tmp_save_path, intersect.astype(np.uint8) * 255)

                # 去除公共区域
                ped_edge_refine = ped_edge * (1-intersect)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, REFINE_ERODE_PED_CROSSING)
                ped_edge_refine = cv2.erode(ped_edge_refine, kernel)

                # 去除杂点
                _, labels, stats, centroids = cv2.connectedComponentsWithStats(ped_edge_refine)
                i = 0
                for istat in stats:
                    # 面积
                    if istat[4] < REMOVE_AREA_PED:
                        if istat[3] > istat[4]:
                            r = istat[3]
                        else:
                            r = istat[4]
                        cv2.rectangle(ped_edge_refine, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26
                    i = i + 1

                # 再次膨胀
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, REFINE_DILATE_PED_CROSSING)
                ped_edge_refine = cv2.dilate(ped_edge_refine, kernel)

                if INCREASE_SPACE:
                    tmp_save_path_refine = os.path.join(otuput_dir_vis, "refine")
                    tmp_save_path = os.path.join(tmp_save_path_refine, "ped_refine.png")
                    cv2.imwrite(tmp_save_path, ped_edge_refine.astype(np.uint8) * 255)

                label_dict["ped_crossing"] = ped_edge_refine.astype(np.uint8)

            # 路沿精细化 去除不存在的边缘线
            if REMOVE_ROAD_EDGE:
                if "drivable_area" in STATIC_LAYER_NAME:

                    if INCREASE_SPACE:
                        tmp_save_path_refine = os.path.join(otuput_dir_vis, "refine")
                        if not os.path.exists(tmp_save_path_refine): os.makedirs(tmp_save_path_refine)

                    # 去除多余线条
                    road_edge = label_dict["drivable_area"]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, REFINE_ERODE_ROAD_EDGE)
                    road_edge = cv2.erode(road_edge, kernel)

                    # 角点检测
                    dst = cv2.cornerHarris(road_edge, HARRIS_NEIGHBOR_SIZE, HARRIS_SOBLE_SIZE, HARRIS_K)
                    mask = (dst > HARRIS_THRESHOLD * np.max(dst))

                    # 去除角点区域
                    index_y, index_x = np.where(mask == True)
                    list_pts = list(zip(index_x, index_y))

                    for pt in list_pts:
                        cv2.circle(road_edge, pt, POINTS_REMOVE_RADIUS, 0, -1)

                    # 去除杂点区域
                    _, labels, stats, centroids = cv2.connectedComponentsWithStats(road_edge)
                    i = 0
                    for istat in stats:
                        # 面积
                        if istat[4] < REMOVE_AREA_ROAD:
                            if istat[3] > istat[4]:
                                r = istat[3]
                            else:
                                r = istat[4]
                            cv2.rectangle(road_edge, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26
                        i = i + 1

                    # 重新进行膨胀
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, REFINE_DILATE_ROAD_EDGE)
                    road_edge = cv2.dilate(road_edge, kernel)

                    if INCREASE_SPACE:
                        tmp_save_path_refine = os.path.join(otuput_dir_vis, "refine")
                        tmp_save_path = os.path.join(tmp_save_path_refine, "road_refine.png")
                        cv2.imwrite(tmp_save_path, road_edge.astype(np.uint8) * 255)

                    label_dict["drivable_area"] = road_edge

        # occlude
        occlude_label_path = os.path.join(output_dir_tmp, "bev_occlusion_mask.npy")
        if USING_OCCLUDE:
            label_dict["occlude"] = np.load(occlude_label_path)

        #---------------------------------------------------#
        #  生成训练标签
        #---------------------------------------------------#
        target_seg_label = np.zeros([nx[0],nx[1]],dtype=np.uint8)
        len_static = len(STATIC_LAYER_NAME)
        len_dynamic = len(DYNAMIC_OBJECT_LIST)

        # 首先加载静态标签
        counter = 0
        for idx, static_name in enumerate(STATIC_LAYER_NAME_SEQUENCE):
            label_static = label_dict[static_name]

            counter+=1
            if static_name=="road_divider":
                counter-=1
            target_seg_label[label_static==1] = counter + len_dynamic

        # 然后加载动态标签
        if DRAW_DYNAMIC:
            for idx,dynamic_name in enumerate(DYNAMIC_OBJECT_LIST):
                instance = label_dict[dynamic_name]["instance"]
                target_seg_label[instance!=0] = idx + 1

        # 然后用occlude去覆盖不能预测的区域
        if USING_OCCLUDE:
            occlude = np.load(occlude_label_path)

            # 进行一定腐蚀操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE_OCCLUSION)
            occlude = cv2.erode(occlude, kernel)
            occlude = cv2.dilate(occlude, kernel)
            target_seg_label[occlude != 0] = IGNORE_INDEX
            # occlusion total mask
            occlude_mask = (occlude != 0)

        if INCREASE_SPACE:
            tmp_save_path = os.path.join(otuput_dir_vis,"final_seg_label.png")
            cv2.imwrite(tmp_save_path, target_seg_label.astype(np.uint8) * BEV_VIS_MULTIPLY)

            target_dir = os.path.join(NUSCENE_DATA, str(INDEX_START) + "_" + str(INDEX_END)+"_final_label")
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            target_path = os.path.join(target_dir, str(tmp_index) + ".png")
            shutil.copy(tmp_save_path,target_path)

        # ---------------------------------------------------#
        #  生成centermap和offsetmap
        # ---------------------------------------------------#
        # 这个是总的center和offset
        center_label = np.zeros((nx[0], nx[1]))
        offset_label = IGNORE_INDEX * np.ones((2, nx[0], nx[1]))
        x, y = torch.meshgrid(torch.arange(int(nx[0]), dtype=torch.float), torch.arange(int(nx[1]), dtype=torch.float))


        for dynamic_name in DYNAMIC_OBJECT_LIST:
            instance_all = label_dict[dynamic_name]["instance"]
            id_list_all = np.array(label_dict[dynamic_name]["id_list"])

            for idx in range(id_list_all.shape[0]):
                tmp_id = int(id_list_all[idx,0])
                tmp_mask = (instance_all==tmp_id)

                if USING_OCCLUDE:
                    judge_mask = np.logical_not(occlude_mask) * tmp_mask
                    # 车子被遮挡
                    if np.all(judge_mask==0):
                        continue

                # 车子没有被遮挡

                # 产生offset图
                xc = x[tmp_mask].mean().round().long()
                yc = y[tmp_mask].mean().round().long()
                off_x = xc - x
                off_y = yc - y
                offset_label[0,tmp_mask] = off_x[tmp_mask]
                offset_label[1,tmp_mask] = off_y[tmp_mask]

                # 产生centerpoint图
                g = np.exp(-(off_x ** 2 + off_y ** 2) / SIGMA ** 2)
                center_label = np.maximum(center_label, g)

        # 转成 numpy
        center_label = center_label

        if INCREASE_SPACE:
            offset_label_path = os.path.join(output_dir_tmp, "bev_dynamic_offset.npy")
            center_label_path = os.path.join(output_dir_tmp, "bev_dynamic_center.npy")
            np.save(offset_label_path,offset_label)
            np.save(center_label_path,center_label)

        if DEBUG:
            plt.imshow(offset_label[0])
            plt.xticks([])  # 去掉x轴
            plt.yticks([])  # 去掉y轴
            plt.axis('off')  # 去掉坐标轴
            plt.savefig(os.path.join(otuput_dir_vis, "bev_dynamic_offset_x.png"), bbox_inches='tight', pad_inches=0)

            plt.imshow(offset_label[1])
            plt.xticks([])  # 去掉x轴
            plt.yticks([])  # 去掉y轴
            plt.axis('off')  # 去掉坐标轴
            plt.savefig(os.path.join(otuput_dir_vis, "bev_dynamic_offset_y.png"), bbox_inches='tight', pad_inches=0)

        if DEBUG:
            plt.imshow(center_label)
            plt.xticks([])  # 去掉x轴
            plt.yticks([])  # 去掉y轴
            plt.axis('off')  # 去掉坐标轴
            plt.savefig(os.path.join(otuput_dir_vis, "bev_dynamic_center.png"), bbox_inches='tight', pad_inches=0)

        # final label保存
        final_label= dict()
        final_label["semantic_seg"] = target_seg_label
        final_label["offset"] = offset_label
        final_label["center"] = center_label

        if USING_LINE:
            final_label_path = os.path.join(output_dir_tmp, "train_label_line.npy")
        else:
            final_label_path = os.path.join(output_dir_tmp, "train_label.npy")

        np.save(final_label_path, final_label)
