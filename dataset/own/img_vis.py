import numpy as np
import cv2
from config_file import *

import imageio

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
    target_fps = 10
    GEN_GIF = True
    GEN_MP4 = True

    # 导出参数
    list_item = os.listdir(root_dir)
    lambda_func = lambda x: int(x)
    list_item.sort(key=lambda_func)
    target_size = (width * 3, height * 2)

    if GEN_MP4:
        output_path = os.path.join(os.path.dirname(root_dir), "syn_data_vis.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, target_size)
    else:
        output_path = os.path.join(os.path.dirname(root_dir),"syn_data_vis.avi")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), target_fps, target_size)


    frames = []
    counter = 0

    for item in list_item:
        counter +=1
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

        out.write(img_draw)

        if GEN_GIF and 50<counter<57:
            cv2.imwrite( os.path.join(tmp_item_path,"stitch.png"), img_draw)
            frames.append(imageio.imread(os.path.join(tmp_item_path,"stitch.png")))

        if counter >=57:
            imageio.mimsave(os.path.join(os.path.dirname(root_dir), "syn_data_vis.gif"), frames, 'GIF',fps= 10)
            exit()


        print("finishing reading image %s" % tmp_item_path)