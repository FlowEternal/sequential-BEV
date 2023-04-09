import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import cv2
import yaml

import multiprocessing
import tqdm

from pyquaternion import Quaternion
import skimage.draw

import warnings
warnings.filterwarnings("ignore")

from dataset.nuscene.Config import *

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def floodFill(px, color, inputImg, outputImg):

    mask = np.zeros((inputImg.shape[0]+2, inputImg.shape[1]+2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(image=inputImg, mask=mask, seedPoint=(px[0], px[1]), newVal=(255,255,255), loDiff=(1,1,1),
                  upDiff=(1,1,1), flags=flags)

    outputImg[np.where(mask[1:-1, 1:-1] == 255)] = color


def castRay(fromPoint, toPoint, inputImg, outputImg):

    # loop over all pixels along the ray, moving outwards
    ray = list(zip(*skimage.draw.line(*(int(fromPoint[0]), int(fromPoint[1])), *(int(toPoint[0]), int(toPoint[1])))))
    stopRay = stopTransfer = False
    for px in ray:

        # out-of-bounds check
        if not (0 <= px[0]  < inputImg.shape[1] and 0 <= px[1] < inputImg.shape[0]):
            continue

        # check if ray hit a blocking object class
        for label in BLOCKING_LABELS:
            if (inputImg[px[1], px[0], :] == COLORS[label]).all():

                # if car, continue ray to look for more blocking objects, else stop ray
                if label == "vehicle":
                    if stopTransfer: # if car behind another car, skip
                        continue
                    else: # if first car in line of ray
                        stopTransfer = True
                else:
                    stopRay = True

                # transfer blocking object to output image
                if not (outputImg[px[1], px[0], :] == COLORS[label]).all():
                    floodFill(px, COLORS[label], inputImg, outputImg)
                break

        if stopRay: # stop ray if blocked
            break
        if stopTransfer: # if transfer is stopped, still look for tall non-blocking labels to transfer
            for label in TALL_NON_BLOCKING_LABELS:
                if (inputImg[px[1], px[0], :] == COLORS[label]).all():
                    outputImg[px[1], px[0], :] = inputImg[px[1], px[0], :]
                    break

        else: # transfer px to output image
            outputImg[px[1], px[0], :] = inputImg[px[1], px[0], :]

class Camera:

    def __init__(self, config, frame, pxPerM):

        self.origin = (frame[0] + config["camX"] * pxPerM[0], frame[1] - config["camY"] * pxPerM[1])

        ## added
        quat = Quaternion([config["cam_quat0"], config["cam_quat1"],
                           config["cam_quat2"], config["cam_quat3"]])

        transf_matrix = quat.transformation_matrix
        self.yaw =  np.arctan2(transf_matrix[1, 0], transf_matrix[0, 0]) /3.14 * 180.0 + 90.0

        self.fov = 2.0 * np.arctan(config["px"] / config["fx"]) * 180.0 / np.pi
        thetaMin = self.yaw - self.fov / 2.0
        thetaMax = (self.yaw + self.fov / 2.0)
        thetaMin = thetaMin % 180 if thetaMin < -180 else thetaMin
        thetaMax = thetaMax % -180 if thetaMax > 180 else thetaMax
        self.fovBounds = (thetaMin, thetaMax)

    def canSee(self, x, y):

        dx, dy = x - self.origin[0], y - self.origin[1]
        theta = np.arctan2(dy, dx) * 180.0 / np.pi
        if self.fovBounds[0] > self.fovBounds[1]:
            return (self.fovBounds[0] <= theta) or (theta <= self.fovBounds[1])
        else:
            return (self.fovBounds[0] <= theta) and (theta <= self.fovBounds[1])

# multiprocess
def process_one_sample(tmp_index):
    print("processing folder %s" % (tmp_index))
    output_dir_tmp = os.path.join(output_dir, str(tmp_index))

    path_tmp = os.path.join(output_dir_tmp,"gt_label/bev_occlusion_mask.npy")
    if os.path.exists(path_tmp):
        print("skip")
        return 0

    process_occlusion_label = np.zeros([nx[0], nx[1]], dtype=np.uint8)

    # ---------------------------------------------------#
    #  静态环境参数
    # ---------------------------------------------------#
    static_dir = os.path.join(output_dir_tmp, "gt_label")
    for static_category, static_color in zip(STATIC_LAYER_NAME, STATIC_LAYER_ID):
        label_path = os.path.join(static_dir, "bev_static_" + static_category + ".npy")
        label_this = np.load(label_path)
        process_occlusion_label[label_this !=0] = static_color

    # ---------------------------------------------------#
    #  动态目标参数
    # ---------------------------------------------------#
    dynamic_dir = os.path.join(output_dir_tmp, "gt_label")
    for dynamic_category, dynamic_color in zip(DYNAMIC_OBJECT_LIST, DYNAMIC_OBJECT_ID):
        seg_numpy = np.load(os.path.join(dynamic_dir, "bev_dynamic_" + dynamic_category + "_seg.npy"))
        process_occlusion_label[seg_numpy != 0] = dynamic_color

    if INCREASE_SPACE:
        cv2.imwrite(os.path.join(output_dir_tmp, "visual", "bev_merge_static_dynamic.png"), process_occlusion_label)

    # ---------------------------------------------------#
    #  翻转label
    # ---------------------------------------------------#
    # 逆时针翻转90度
    process_occlusion_label = cv2.flip(process_occlusion_label, 1)
    inputImg = cv2.cvtColor(cv2.transpose(process_occlusion_label), cv2.COLOR_GRAY2BGR)

    pt1 = (base_link[0] - M1, base_link[1] - M2)
    pt2 = (base_link[0] + M1, base_link[1] + M2)
    cv2.rectangle(inputImg, pt1, pt2, COLORS["vehicle"], -1)

    # ---------------------------------------------------#
    #  遮挡区域去除
    # ---------------------------------------------------#
    cam_config_list = []
    for cam_name in CAMERA:
        cam_xml_dir = os.path.join(output_dir_tmp, "cameras", cam_name + ".yaml")
        cam_config = yaml.safe_load(open(cam_xml_dir))
        cam = Camera(cam_config, base_link, pxPerM)
        cam_config_list.append(cam)

    outputImg = np.zeros(inputImg.shape, dtype=np.uint8) + np.array(COLORS["occluded"], dtype=np.uint8)

    # temporarily recolor ego vehicle (if in image), s.t. it does not block
    if base_link[0] > 0 and base_link[1] > 0:
        floodFill(base_link, DUMMY_COLOR, inputImg, inputImg)

    # loop over all border pixels to determine if ray is visible
    rays = []
    for cam in cam_config_list:
        for x in range(inputImg.shape[1]):
            if cam.canSee(x, 0):
                rays.append((cam.origin, (x, 0)))
        for x in range(inputImg.shape[1]):
            if cam.canSee(x, inputImg.shape[0]):
                rays.append((cam.origin, (x, inputImg.shape[0])))
        for y in range(inputImg.shape[0]):
            if cam.canSee(0, y):
                rays.append((cam.origin, (0, y)))
        for y in range(inputImg.shape[0]):
            if cam.canSee(inputImg.shape[1], y):
                rays.append((cam.origin, (inputImg.shape[1], y)))

    # cast rays
    for ray in rays:
        castRay(ray[0], ray[1], inputImg, outputImg)

    # recolor ego vehicle as car and transfer to output
    if base_link[0] > 0 and base_link[1] > 0:
        floodFill(base_link, COLORS["vehicle"], inputImg, outputImg)
        floodFill(base_link, COLORS["vehicle"], inputImg, outputImg)

    # ---------------------------------------------------#
    #  保存图片
    # ---------------------------------------------------#
    # 顺时针翻转90度
    outputImg = cv2.flip(outputImg, 1)
    outputImg = cv2.transpose(outputImg)
    outputImg = cv2.flip(outputImg, 1)
    outputImg = cv2.transpose(outputImg)
    outputImg = cv2.flip(outputImg, 1)
    outputImg = cv2.transpose(outputImg)

    occlude_mask = (outputImg[:, :, 0] == int(OCCLUSION_COLOR)).astype(np.uint8)

    if INCREASE_SPACE:
        # 保存可视化
        cv2.imwrite(os.path.join(output_dir_tmp, "visual", "bev_merge_static_dynamic_occluded.png"), outputImg)
        # 保存可视化occlusion mask
        cv2.imwrite(os.path.join(output_dir_tmp, "visual", "bev_occlusion_mask.png"), occlude_mask * 255)

    # 保存numpy矩阵
    np.save(os.path.join(output_dir_tmp, "gt_label", "bev_occlusion_mask.npy"), occlude_mask)

#---------------------------------------------------#
#  导出参数
#---------------------------------------------------#
# 数据的根目录
output_dir = os.path.join(NUSCENE_DATA,str(INDEX_START) + "_" + str(INDEX_END))
if not os.path.exists(output_dir):os.makedirs(output_dir)

#---------------------------------------------------#
#  尺寸计算
#---------------------------------------------------#
dx, bx, nx = gen_dx_bx(XBOUND, YBOUND, ZBOUND)
dx, bx = dx[:2].numpy(), bx[:2].numpy()

pxPerM = ( float(nx[0]) / (XBOUND[1] - XBOUND[0]), float(nx[1]) / (YBOUND[1] - YBOUND[0]))
base_link = (int(nx[0] / 2.0), int(nx[1]/ 2.0))

M1 = int((CAR_LENGTH / 2.0 + 0.1) * pxPerM[0])
M2 = int((CAR_WIDTH / 2.0 + 0.1) * pxPerM[1])

#---------------------------------------------------#
#  COLOR设置
#---------------------------------------------------#
DUMMY_COLOR = (111,111,111)
COLORS = dict()
for key,value in zip(DYNAMIC_OBJECT_LIST,DYNAMIC_OBJECT_ID):
    COLORS[key] = (value,value,value)
COLORS["occluded"] = (OCCLUSION_COLOR,OCCLUSION_COLOR,OCCLUSION_COLOR)

# ---------------------------------------------------#
#  处理occlusion
# ---------------------------------------------------#

if MULTITHREAD:
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    index_array = list(range(INDEX_START,INDEX_END+1))
    for _ in tqdm.tqdm(pool.imap(process_one_sample, index_array),
                       desc="Processing images",
                       total=len(index_array),
                       smoothing=0):

        pass
    pool.close()
    pool.join()

else:
    for tmp_index in range(INDEX_START, INDEX_END+1):
        process_one_sample(tmp_index)
