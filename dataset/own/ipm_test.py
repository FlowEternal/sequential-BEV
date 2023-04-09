import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")
from config_file import *

def get_rot_matrix_euler(thetaX, thetaY, thetaZ):
  Rz = np.array([[np.cos(-thetaZ), -np.sin(-thetaZ), 0.0], [np.sin(-thetaZ), np.cos(-thetaZ), 0.0], [0.0, 0.0, 1.0]])  # Z
  Ry = np.array([[np.cos(-thetaY), 0.0, np.sin(-thetaY)], [0.0, 1.0, 0.0], [-np.sin(-thetaY), 0.0, np.cos(-thetaY)]])  # Y
  Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-thetaX), -np.sin(-thetaX)], [0.0, np.sin(-thetaX), np.cos(-thetaX)]])  # X
  Rotation_Matrix_Euler = Rx.dot(Ry.dot(Rz))
  return Rotation_Matrix_Euler

if __name__ == '__main__':
  # 设定参数
  # img_name = "img_front.jpg"
  # img_name = "img_back.jpg"

  # img_name = "front_middle.jpg"
  img_name = "front_left.jpg"
  # img_name = "front_right.jpg"
  # img_name = "rear_middle.jpg"
  # img_name = "rear_left.jpg"
  # img_name = "rear_right.jpg"

  wm = 40    # 鸟瞰图宽度(m)
  hm = 80    # 鸟瞰图高度(m)
  res = 10   # 分辨率 pixel/m
  org_size = (1920, 1080)
  interpMode = cv2.INTER_NEAREST # 插值模式

  # default front middle
  (FocalLengthX, FocalLengthY, PrincipalPointX, PrincipalPointY, CamX, CamY, CamZ, thetaX, thetaY,thetaZ) = get_parameter_one(img_name.split(".")[0])


  #---------------------------------------------------#
  #  进行处理
  #---------------------------------------------------#
  # 内参
  K = np.zeros([3, 3])
  K[0, 0] = FocalLengthX
  K[1, 1] = FocalLengthY
  K[0, 2] = PrincipalPointX
  K[1, 2] = PrincipalPointY
  K[2, 2] = 1.0
  CamX = CamX / 1000
  CamY = CamY / 1000
  CamZ = CamZ / 1000

  # front from vehicle to camera
  # R = np.array([ [-0.0277381453893765,-0.999518630548261,0.0138961317382495],
  #                [0.0185641847046977,-0.0144141677104233,-0.999723763254363],
  #                [0.999442827947956,-0.0274725127380944,0.0189550707529030] ] )

  R = get_rot_matrix_euler(thetaX,thetaY, thetaZ)
  print(R)
  X = np.array([ CamX, CamY, CamZ ])

  # 导出参数
  t = -R.dot(X)
  Rt = np.zeros([3, 4])
  Rt[0:3, 0:3] = R
  Rt[0:3, 3] = t
  P = K.dot(Rt)

  #---------------------------------------------------#
  #  计算运算矩阵
  #---------------------------------------------------#
  pxPerM = (res, res)
  outputRes = (int(wm * pxPerM[0]), int(hm * pxPerM[1]))

  # setup mapping from street/top-image plane to world coords
  shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
  M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]],
                [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]])

  # find IPM as inverse of P*M
  transfer_matrix = np.linalg.inv(P.dot(M))
  mask = np.zeros((outputRes[0], outputRes[1]), dtype=np.bool)
  yaw_plan = 90 + np.rad2deg(np.arctan2( np.linalg.inv(R)[1, 0], np.linalg.inv(R)[0, 0] ) )
  for i in range(outputRes[1]):
     for j in range(outputRes[0]):
      theta = np.rad2deg(np.arctan2(-j + outputRes[0]/2 , i - outputRes[1] / 2 ))
      if  90 < abs(theta - yaw_plan) < 270:
        mask[j,i] = True

  img = cv2.resize(cv2.imread(os.path.join("test_imgs",img_name)), org_size)
  processed_img = cv2.warpPerspective(img, transfer_matrix, (outputRes[1], outputRes[0]), flags=interpMode)
  processed_img[mask] = 0
  cv2.imwrite(os.path.join("test_imgs_out", "bev_" + img_name) ,processed_img)
  exit()



