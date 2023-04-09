import os
import shutil
import numpy as np

def recursive_remove(root_dir):
    tmp_list = os.listdir(root_dir)
    for item in tmp_list:
        os.remove(os.path.join(root_dir, item))
    os.removedirs(root_dir)

if __name__ == '__main__':
    # 设定参数
    cam_index = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"]
    root_dir = "D:\\Data\\BEV\\own\\pack15"
    data_start_time = "2021_07_08_14_18_18_482"
    # TIME_INTERVAL = 0.035 # 35ms
    TIME_INTERVAL = 0.050 # 35ms

    # 导出参数
    CAMERA = [data_start_time + "_" + cam for cam in cam_index]
    syn_file_dir = os.path.join(root_dir, "syn_data")
    if not os.path.exists(syn_file_dir): os.makedirs(syn_file_dir)

    # 取最小的为base
    list_all = [ os.listdir(os.path.join(root_dir, cam)) for cam in CAMERA]
    min_num = np.array([len(tmp_list) for tmp_list in list_all]).min()
    min_index = np.array([len(tmp_list) for tmp_list in list_all]).argmin()
    print("base camera:")
    print(CAMERA[min_index])
    base_list = list_all[min_index]

    # 开始软同步
    time_stamp=0
    for tmp_name in base_list:
        ref_time = float(tmp_name.split("_")[0])
        # print(ref_time)
        time_stamp+=1
        tmp_save_file = os.path.join(syn_file_dir,str(time_stamp))
        if not os.path.exists(tmp_save_file):
            os.makedirs(tmp_save_file)
        else:
            continue

        src_dir_ = os.path.join(root_dir, CAMERA[min_index],tmp_name)
        dst_dir_ = os.path.join(tmp_save_file,cam_index[min_index] +"_"+tmp_name)
        shutil.copy(src_dir_, dst_dir_)

        for index in range(0,len(list_all)):
            if index == min_index:
                continue

            cmp_list = list_all[index]
            cmp_time = np.array([float(tmp_name.split("_")[0]) for tmp_name in cmp_list])
            diff_time = np.abs(cmp_time - ref_time)
            valid_index_array = np.where((diff_time < TIME_INTERVAL))

            if len(valid_index_array[0]) !=1:
                print(str(time_stamp) + " can not syn")
                recursive_remove(tmp_save_file)
                break

            valid_index = int(valid_index_array[0])
            valid_file = cmp_list[valid_index]
            src_dir = os.path.join(root_dir, CAMERA[index],valid_file)
            dst_dir = os.path.join(tmp_save_file,cam_index[index] +"_"+valid_file)
            shutil.copy(src_dir, dst_dir)

