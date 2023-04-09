import os
import random

if __name__ == '__main__':
    # 439 - 479有问题 过滤掉
    # main_dir = "/data/zdx/Data/data_nuscene"
    # set_name = "0_19"
    # train_ratio = 0.90
    # train_name = "train_mini.txt"
    # valid_name = "valid_mini.txt"

    main_dir = "/data/zdx/Data/data_nuscene"
    set_name = "0_34148"
    train_ratio = 0.998
    train_name = "train.txt"
    valid_name = "valid.txt"

    root_dir = os.path.join(main_dir, set_name)
    train_txt = open(os.path.join(main_dir,train_name),"w")
    val_txt = open(os.path.join(main_dir,valid_name),"w")

    list_items = os.listdir(root_dir)
    random.shuffle(list_items)
    items_len = len(list_items)
    train_len = int(items_len * train_ratio)
    train_list = list_items[0: train_len]
    val_list = list_items[train_len: ]

    for name in train_list:
        if int(name) not in list(range(439, 480)):
            train_txt.writelines( os.path.join(root_dir,name) + "\n")
        else:
            print("skip train")
    for name in val_list:
        if int(name) not in list(range(439, 480)):
            val_txt.writelines( os.path.join(root_dir,name) + "\n")
        else:
            print("skip valid")