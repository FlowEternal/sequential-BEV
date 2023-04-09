import cv2
import os
import numpy as np
import webcolors

STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result

idx_choose = 0
label_dir = "D:\\Data\\BEV\\own\\pack3\\syn_data_label"
offset_height = 360 * 2
label_list = os.listdir(label_dir)
choose_label = label_list[idx_choose]

# read label
# type_class_list = ["crosswalk","edge","lane"]
type_class_list = ["edge"]

type_class_color = [from_colorname_to_bgr('BlueViolet'),from_colorname_to_bgr('Red'),from_colorname_to_bgr('Salmon')]
type_dict = {}
for (key, value) in zip(type_class_list, type_class_color):
    type_dict[key] = value

print(type_class_color)

import json
label_path = os.path.join(label_dir, choose_label)
label_info = json.load(open(label_path,"r"))["shapes"]

bev_label_draw = np.zeros((960, 1920, 3),dtype=np.uint8)

for target_label in type_class_list:
    for one_label in label_info:
        type_name = one_label["label"]
        if type_name != target_label:
            continue
        tmp_points = np.array(one_label["points"])
        tmp_points[:,1] = tmp_points[:,1] - offset_height
        tmp_points = tmp_points.astype(np.int32)

        for idx in range(tmp_points.shape[0]-1):
            pt1 = (int(tmp_points[idx][0]), int(tmp_points[idx][1]))
            pt2 = (int(tmp_points[idx+1][0]), int(tmp_points[idx+1][1]))
            cv2.line(bev_label_draw, pt1, pt2, type_dict[type_name], 18)

bev_label_draw = cv2.resize(bev_label_draw,(80 * 10, 40 * 10))
cv2.imwrite("label_%i.png" %idx_choose, bev_label_draw)
