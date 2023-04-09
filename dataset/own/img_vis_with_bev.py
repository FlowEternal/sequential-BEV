import os
import numpy as np
import cv2
from config_file import *

DELTA_X = 50
DELTA_Y = 80
txt_org = (0 + DELTA_X, 0 + DELTA_Y)
font_scale = 2
thickness = 4
color = (0, 0, 255)

if __name__ == '__main__':
    # 设定参数
    root_dir = os.path.join(DATA_ROOT_DIR,"syn_data")
    height = 360
    width = 640
    target_fps = 20

    # 导出参数
    list_item = os.listdir(root_dir)
    lambda_func = lambda x: int(x)
    list_item.sort(key=lambda_func)
    target_size = (width * 3, height * 2 + int(width * 3 / 2) )
    output_path = os.path.join(os.path.dirname(root_dir),"syn_data_vis_bev.avi")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), target_fps, target_size)

    for item in list_item:
        tmp_item_path = os.path.join(root_dir, item)
        tmp_item = os.listdir(tmp_item_path)

        img_draw = np.zeros([target_size[1], target_size[0], 3], dtype=np.uint8)
        for idx, (key, value) in enumerate(mapping_cam.items()):
            rol_idx = idx % 3
            col_idx = int(idx / 3)

            img_path = ""
            for tmp_path in tmp_item:
                if value in tmp_path:
                    img_path = os.path.join(tmp_item_path,tmp_path)
                    break
            img_tmp = cv2.resize(cv2.imread(img_path),(width,height))
            img_tmp_ = cv2.putText(img_tmp,key,txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
            img_draw[height*col_idx:height*(col_idx + 1),width*rol_idx:width*(rol_idx + 1) ,:] = img_tmp_

        bev_path = ""
        for tmp_name in tmp_item:
            if "bev_stitch.png" in tmp_name:
                bev_path = os.path.join(tmp_item_path, tmp_name)
                break

        bev_img = cv2.resize(cv2.imread(bev_path), ( width * 3, int(width * 3 / 2)) )
        img_draw[height * 2:, :, :] = bev_img

        out.write(img_draw)
        print("finishing reading image %s" % tmp_item_path)