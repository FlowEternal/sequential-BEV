import os
import random
random.seed(19911220)
from nuscenes.nuscenes import NuScenes

if __name__ == '__main__':
    #---------------------------------------------------#
    #  数据集设定
    #---------------------------------------------------#
    SEQUENCE_LENGTH = 3
    train_ratio = 0.998
    # train_ratio = 0.60
    train_name = "train_seq%i.txt" %SEQUENCE_LENGTH
    valid_name = "valid_seq%i.txt" %SEQUENCE_LENGTH

    # SET_NAME = "0_19"
    # INDEX_START = 0
    # INDEX_END = 19
    # VERSION = "mini"
    # NUSCENE_DATA = "/data/zdx/Data/data_nuscene"

    SET_NAME = "0_34148"
    INDEX_START = 0
    INDEX_END = 34148
    VERSION = "trainval"
    NUSCENE_DATA = "/data/zdx/Data/data_nuscene"

    #---------------------------------------------------#
    #  导出参数
    #---------------------------------------------------#
    # 数据的根目录
    output_dir = os.path.join(NUSCENE_DATA,str(INDEX_START) + "_" + str(INDEX_END))
    if not os.path.exists(output_dir):os.makedirs(output_dir)

    # nuscene对象
    nusc = NuScenes(version='v1.0-{}'.format(VERSION),dataroot=os.path.join(NUSCENE_DATA, VERSION),verbose=True)
    samples = nusc.sample
    print(f"total sample number is {len(samples)}")

    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    scene_token_dict = {}
    #---------------------------------------------------#
    #  离线提取数据
    #---------------------------------------------------#
    for tmp_index in range(INDEX_START, INDEX_END+1):
        tmp_sample = samples[tmp_index]
        scene_token = tmp_sample['scene_token']

        if scene_token_dict.get(scene_token) is None:
            scene_token_dict.update({scene_token:[tmp_index]})
        else:
            scene_token_dict[scene_token].append(tmp_index)


    for key, value in scene_token_dict.items():
        print(value[-1])

    shuffle_list = []
    for key, value in scene_token_dict.items():
        print("processing scene: " + key + " with length %i" % len(value))
        print(value)

        for id_seq in range(0, len(value)- (SEQUENCE_LENGTH -1)):
            shuffle_list.append(value[id_seq: id_seq + SEQUENCE_LENGTH])

    print(shuffle_list)
    random.shuffle(shuffle_list)
    print(shuffle_list)

    train_txt = open(os.path.join(NUSCENE_DATA,train_name),"w")
    val_txt = open(os.path.join(NUSCENE_DATA,valid_name),"w")

    items_len = len(shuffle_list)
    train_len = int(items_len * train_ratio)
    train_list = shuffle_list[0: train_len]
    val_list = shuffle_list[train_len: ]

    for tmp_array in train_list:
        filter_flag = False
        for tmp_id in tmp_array:
            if tmp_array in list(range(439, 480)):
                filter_flag = True
                break
        if not filter_flag:
            for tmp_id in tmp_array:
                train_txt.writelines( os.path.join(NUSCENE_DATA, SET_NAME,str(tmp_id) ) + " ")
            train_txt.writelines("\n")
        else:
            print("skip train")

    for tmp_array in val_list:
        filter_flag = False
        for tmp_id in tmp_array:
            if tmp_array in list(range(439, 480)):
                filter_flag = True
                break
        if not filter_flag:
            for tmp_id in tmp_array:
                val_txt.writelines(os.path.join(NUSCENE_DATA, SET_NAME, str(tmp_id)) + " ")
            val_txt.writelines("\n")
        else:
            print("skip train")
