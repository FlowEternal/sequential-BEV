import os
import glob


if __name__ == '__main__':
    # parameter setting
    img_dir = "D:\\Data\\BEV\\own\\pack1_image"
    img_list = glob.glob(os.path.join(img_dir,"*.jpg"))

    for img_path_ in img_list:
        name_idx = img_path_[-9:]
        replace = name_idx[1:-4].zfill(9)
        img_path_new = img_path_[0:-9] + "1_%s.jpg" % replace
        print(img_path_new)
        os.rename(img_path_, img_path_new)