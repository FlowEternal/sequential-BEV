"""
Function: Generate Calibration Tensor For c++ Implementation
Author: Zhan Dong Xu
Date: 2021/11/11
"""

import numpy as np
matrix_path = "../../data/matrix_nuscene.npy"
name_list = ["front_mat","front_left_mat","front_right_mat","rear_mat", "rear_left_mat", "rear_right_mat"]

pm_tensor = np.load(matrix_path)
for idx in range(len(name_list)):
    pm_mat = pm_tensor[idx].reshape(-1)
    name_ = name_list[idx]

    str_info = "float " + name_ + "[9] = {"
    for tmp_number in pm_mat:
        str_info+=str(tmp_number)+","

    str_info = str_info[0:-1] + "};"
    print(str_info)



