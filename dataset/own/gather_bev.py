import os
import glob
import shutil

root_dir = "D:\\Data\\bev"
data_dir = os.path.join(root_dir, "syn_data")
target_dir = os.path.join(root_dir, "bev_data")
if not os.path.exists(target_dir): os.makedirs(target_dir)

bev_file_list = glob.glob(os.path.join(data_dir,"*/cam2*.jpg"))
for tmp_file in bev_file_list:
    src_dir = tmp_file
    # id = tmp_file.split("/")[-2]
    # dst_dir = os.path.join(target_dir, id)
    dst_dir = target_dir

    if not os.path.exists(dst_dir): os.makedirs(dst_dir)
    # dst_img = os.path.join(dst_dir,"bev_stitch.png")
    dst_img = os.path.join(dst_dir, os.path.basename(tmp_file))

    shutil.copy(src_dir,dst_img)