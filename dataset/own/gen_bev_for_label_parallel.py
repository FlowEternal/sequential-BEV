import numpy as np
import cv2
import warnings
import multiprocessing
import tqdm

warnings.filterwarnings("ignore")
from config_file import *

def get_rot_matrix_euler(thetaX, thetaY, thetaZ):
  Rz = np.array([[np.cos(-thetaZ), -np.sin(-thetaZ), 0.0], [np.sin(-thetaZ), np.cos(-thetaZ), 0.0], [0.0, 0.0, 1.0]])  # Z
  Ry = np.array([[np.cos(-thetaY), 0.0, np.sin(-thetaY)], [0.0, 1.0, 0.0], [-np.sin(-thetaY), 0.0, np.cos(-thetaY)]])  # Y
  Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-thetaX), -np.sin(-thetaX)], [0.0, np.sin(-thetaX), np.cos(-thetaX)]])  # X
  Rotation_Matrix_Euler = Rx.dot(Ry.dot(Rz))
  return Rotation_Matrix_Euler

def process_one_sample(item):
    tmp_item_path = os.path.join(root_dir, item)
    tmp_item = os.listdir(tmp_item_path)
    processed_imgs = []

    # 循环每一个图片
    for idx, (key, value) in enumerate(mapping_cam.items()):
        img_path = ""
        for tmp_path in tmp_item:
            if value in tmp_path:
                img_path = os.path.join(tmp_item_path, tmp_path)
                break

        org_img = cv2.imread(img_path)
        R_tmp = dict_param[idx][0]
        transfer_matrix_tmp = dict_param[idx][1]
        mask = np.zeros((outputRes[0], outputRes[1]), dtype=np.bool)
        yaw_plan = 90 + np.rad2deg(np.arctan2(np.linalg.inv(R_tmp)[1, 0], np.linalg.inv(R_tmp)[0, 0]))
        for i in range(outputRes[1]):
            for j in range(outputRes[0]):
                theta = np.rad2deg(np.arctan2(-j + outputRes[0] / 2, i - outputRes[1] / 2))
                if 90 < abs(theta - yaw_plan) < 270:
                    mask[j, i] = True

        processed_img = cv2.warpPerspective(org_img, transfer_matrix_tmp, (outputRes[1], outputRes[0]),
                                            flags=interpMode)
        processed_img[mask] = 0
        processed_imgs.append(processed_img)
        # if idx == 0:
        #     cv2.imwrite(os.path.join(tmp_item_path, "front_left_ipm.png"), processed_img)

    # stitch separate images to total bird's-eye-view
    birdsEyeView = np.zeros(processed_imgs[0].shape, dtype=np.uint8)
    for warpedImg in processed_imgs:
        mask = np.any(warpedImg != 0, axis=-1)
        birdsEyeView[mask] = warpedImg[mask]

    cv2.imwrite(os.path.join(tmp_item_path, "bev_stitch.png"), birdsEyeView)
    print("finishing reading image %s" % tmp_item_path)

if __name__ == '__main__':
    # 设定参数
    wm = 40  # 鸟瞰图宽度(m)
    hm = 80  # 鸟瞰图高度(m)
    res = 10  # 分辨率 pixel/m
    interpMode = cv2.INTER_NEAREST  # 插值模式
    root_dir = os.path.join(DATA_ROOT_DIR,"syn_data")

    list_item = os.listdir(root_dir)
    lambda_func = lambda x: int(x)
    list_item.sort(key=lambda_func)

    # ---------------------------------------------------#
    #  计算运算矩阵
    # ---------------------------------------------------#
    pxPerM = (res, res)
    outputRes = (int(wm * pxPerM[0]), int(hm * pxPerM[1]))

    # setup mapping from street/top-image plane to world coords
    shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
    M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]],
                  [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])

    # ---------------------------------------------------#
    #  计算变换矩阵
    # ---------------------------------------------------#
    dict_param = list()
    for key, value in mapping_cam.items():
        (FocalLengthX,FocalLengthY,PrincipalPointX,PrincipalPointY,CamX,CamY,CamZ,thetaX,thetaY,thetaZ) = func_get_parameter(key)

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

        R = get_rot_matrix_euler(thetaX,thetaY, thetaZ)
        X = np.array([ CamX, CamY, CamZ ])

        # 导出参数
        t = -R.dot(X)
        Rt = np.zeros([3, 4])
        Rt[0:3, 0:3] = R
        Rt[0:3, 3] = t
        P = K.dot(Rt)
        TRANSFER_MATRIX = np.linalg.inv(P.dot(M))
        dict_param.append((R, TRANSFER_MATRIX))

    # 进行ipm变换
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm.tqdm(pool.imap(process_one_sample, list_item),
                       desc="Processing images",
                       total=len(list_item),
                       smoothing=0):
        pass
    pool.close()
    pool.join()
