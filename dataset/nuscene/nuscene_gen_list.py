import os
import random
random.seed(1991)

if __name__ == '__main__':
    #---------------------------------------------------#
    #  输入参数
    #---------------------------------------------------#
    TRAIN_RATIO = 0.90

    # ROOT_DIR_LIST = ["/data_2/zhandongxu/nuscene/0_19"]
    # OUTPUT_TRAIN_TXT = "/home/zhandongxu/Code/bev_static/data/train_list_debug.txt"
    # OUTPUT_VALID_TXT = "/home/zhandongxu/Code/bev_static/data/valid_list_debug.txt"

    # ROOT_DIR_LIST = ["/data_2/zhandongxu/nuscene/0_34148"]
    # OUTPUT_TRAIN_TXT = "/home/zhandongxu/Code/bev_static/data/train_list_run.txt"
    # OUTPUT_VALID_TXT = "/home/zhandongxu/Code/bev_static/data/valid_list_run.txt"

    ROOT_DIR_LIST = ["D:\\AutoDrive\\Data\\nuscene\\0_19"]
    OUTPUT_TRAIN_TXT = "D:\\AutoDrive\\Data\\nuscene\\train_list.txt"
    OUTPUT_VALID_TXT = "D:\\AutoDrive\\Data\\nuscene\\valid_list.txt"

    #---------------------------------------------------#
    #  开始生成
    #---------------------------------------------------#
    total_list = []
    for root_dir in ROOT_DIR_LIST:
        total_list+=[os.path.join(root_dir,folder_name)  for folder_name in os.listdir(root_dir)]

    # total_list = total_list[0:1000]
    # total_list = total_list[5405:]

    total_length = len(total_list)
    train_length = int(total_length * TRAIN_RATIO)
    valid_length = total_length - int(total_length * TRAIN_RATIO)
    print("train example: %i valid example: %i" %(train_length, valid_length))

    #---------------------------------------------------#
    #  写入文本
    #---------------------------------------------------#
    random.shuffle(total_list)

    # train
    train_list = total_list[0:train_length]
    with open(OUTPUT_TRAIN_TXT,"w") as train_writer:
        for rec in train_list:
            train_writer.writelines(rec+"\n")

    # valid
    valid_list = total_list[train_length:]
    with open(OUTPUT_VALID_TXT, "w") as valid_writer:
        for rec in valid_list:
            valid_writer.writelines(rec + "\n")
