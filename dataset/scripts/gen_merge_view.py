import os
import cv2
import numpy as np

video_root = "D:\\Data\\BEV\\own\\pack3"
video_output = video_root + "\\surround_two_vis.mp4"
cam_list = ["FRONT_TWO_vis","LEFT_FRONT_TWO_vis","RIGHT_FRONT_TWO_vis","BACK_TWO_vis","LEFT_BACK_TWO_vis","RIGHT_BACK_TWO_vis"]
video_list = [os.path.join(video_root, cam + ".avi") for cam in cam_list]
vid_list = []
for video_path in video_list:
    vid = cv2.VideoCapture(video_path)
    vid_list.append(vid)



codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter(video_output, codec, 20, (320*3 ,480))

counter = 0
while True:
    vis_img_list = list()
    for vid in vid_list:
        _, input_img = vid.read()
        if input_img is None:
            exit()

        # ---------------------------------------------------#
        #  preprocess
        # ---------------------------------------------------#
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320,240))
        vis_img_list.append(img)

    counter+=1
    img_draw = np.zeros([480, 320 * 3 , 3]).astype(np.uint8)
    height = 240
    width = 320
    DELTA_X = 25
    DELTA_Y = 40
    txt_org = (0 + DELTA_X, 0 + DELTA_Y)
    font_scale = 1
    thickness = 2
    color = (0, 255, 255)
    M1 = 7
    M2 = 14

    # front
    img_front = cv2.putText(vis_img_list[0], "front", txt_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    img_draw[0:height, width: 2 * width, :] = cv2.resize(img_front, (width, height))

    # front left
    img_front_left = cv2.putText(vis_img_list[1], "front left", txt_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                                 thickness)
    img_draw[0:height, 0:width, :] = cv2.resize(img_front_left, (width, height))

    # front right
    img_front_right = cv2.putText(vis_img_list[2], "front right", txt_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                                  thickness)
    img_draw[0:height, 2 * width: 3 * width, :] = cv2.resize(img_front_right, (width, height))

    # back
    img_back = cv2.putText(vis_img_list[3], "back", txt_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    img_draw[height:2 * height, width: 2 * width, :] = cv2.resize(img_back, (width, height))

    # back left
    img_back_left = cv2.putText(vis_img_list[4], "back left", txt_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                                thickness)
    img_draw[height:2 * height, 0: width, :] = cv2.resize(img_back_left, (width, height))

    # back right
    img_back_right = cv2.putText(vis_img_list[5], "back right", txt_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                                 thickness)
    img_draw[height:2 * height, 2 * width: 3 * width, :] = cv2.resize(img_back_right, (width, height))

    video_writer.write(img_draw)

    if counter > 500:
        break