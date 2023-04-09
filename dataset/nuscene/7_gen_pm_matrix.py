import numpy as np
import os,yaml
from pyquaternion import Quaternion

import warnings
warnings.filterwarnings("ignore")
from dataset.nuscene.Config import *

class Camera:
  K = np.zeros([3, 3])
  R = np.zeros([3, 3])
  t = np.zeros([3, 1])
  P = np.zeros([3, 4])

  def setK(self, fx, fy, px, py):
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

  def setR(self, y, p, r):
    # 注意所有的变换的角度都是相对于固定参考坐标系，是extrinsic变换，不是intrinsic变换，这一点很重要
    # 变换是按照 yaw / pitch / roll / translate 的顺序进行
    Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])  # yaw
    Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])  # pitch
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])  # roll
    self.R = Rz.dot(Ry.dot(Rx))

  def setT(self, XCam, YCam, ZCam):
    X = np.array([XCam, YCam, ZCam])
    self.t = -self.R.dot(X)

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)

  def __init__(self, config):
    self.setK(config["fx"], config["fy"], config["px"], config["py"])
    self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
    self.setT(config["camX"], config["camY"], config["camZ"])
    self.updateP()


if __name__ == '__main__':
    res_x = int(1/XBOUND[2])
    res_y = int(1/YBOUND[2])
    bev_width = int((XBOUND[1] - XBOUND[0]) / (XBOUND[2]))
    bev_height = int((YBOUND[1] - YBOUND[0]) / (YBOUND[2]))

    # 数据的根目录
    output_dir = os.path.join(NUSCENE_DATA,str(INDEX_START) + "_" + str(INDEX_END))
    if not os.path.exists(output_dir):os.makedirs(output_dir)


    for tmp_index in range(INDEX_START, INDEX_END + 1):
        print("processing folder %s" % tmp_index)
        output_dir_tmp = os.path.join(output_dir, str(tmp_index))
        camera_dir = os.path.join(output_dir_tmp, "cameras")

        # 产生pm tensor
        camera_configs = []
        cams = []
        for cam_name in CAMERA:
            config_file = os.path.join(camera_dir, cam_name + ".yaml")
            camera_config = yaml.safe_load(open(config_file))

            quat = Quaternion([camera_config["cam_quat0"], camera_config["cam_quat1"],
                               camera_config["cam_quat2"], camera_config["cam_quat3"]])

            yaw, pitch, roll = quat.yaw_pitch_roll
            camera_config["yaw"] = float(yaw) * 180.0 / 3.1415926
            camera_config["pitch"] = float(pitch) * 180.0 / 3.1415926
            camera_config["roll"] = float(roll) * 180.0 / 3.1415926
            camera_configs.append(camera_config)
            cams.append(Camera(camera_config))

        #---------------------------------------------------#
        #  计算运算矩阵
        #---------------------------------------------------#
        # calculate output shape; adjust to match drone image, if specified
        pxPerM = (res_x, res_y)
        outputRes = (bev_width, bev_height)

        # setup mapping from street/top-image plane to world coords
        shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
        M = np.array([[1.0 / pxPerM[0], 0.0, -shift[0] / pxPerM[0]],
                    [0.0, -1.0 / pxPerM[1], shift[1] / pxPerM[1]],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])

        # find IPM as inverse of P*M
        PMs = []
        for cam in cams:
            PMs.append(cam.P.dot(M))

        pm_tensor = np.stack(PMs)
        print(pm_tensor.shape)
        np.save(os.path.join(camera_dir,"pm_tensor.npy"), pm_tensor)

        #        y
        #        |
        #        |
        #        |
        # bev方向 -----------------> x
        # bev uv和原图uv定义一致
        #





