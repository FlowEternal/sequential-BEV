import yaml, os, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1991)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch.utils.data.dataloader
from bev_model import VectorBEV

def deparallel_model(dict_param):
    ck_dict_new = dict()
    for key, value in dict_param.items():
        temp_list = key.split(".")[1:]
        new_key = ""
        for tmp in temp_list:
            new_key += tmp + "."
        ck_dict_new[new_key[0:-1]] = value
    return ck_dict_new

if __name__ == '__main__':
    # https://www.cnblogs.com/jiangkejie/p/13256094.html
    #---------------------------------------------------#
    #  参数设定
    #---------------------------------------------------#
    # log_path = "logs/stn_seq1_finetune"
    # model_name = "epoch_11.pth"

    log_path = "logs/stn_seq2_finetune"
    model_name = "epoch_11.pth"

    # log_path = "logs/stn_seq3_finetune"
    # model_name = "epoch_11.pth"

    video_root = "data/video_nuscene"
    mask_name = "data/mask_nuscene.npy"

    # 导出参数
    cfg_path = os.path.join(log_path, "config.yml")
    cfgs = yaml.safe_load(open(cfg_path))
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    seq_num = cfgs["dataloader"]["seq_num"]
    cam_list = cfgs["dataloader"]["camera_list"]

    net_input_size = (net_input_width, net_input_height)
    use_distribute = cfgs["train"]["use_distribute"]

    # for transfomer encoder - decoder
    layer_id = cfgs["backbone"]["layer_id"]
    use_transformer_decoder = cfgs["backbone"]["use_transformer_decoder"]
    feat_height = int(net_input_height / pow(2, layer_id + 1))
    feat_width = int(net_input_width / pow(2, layer_id + 1))

    x_bound = cfgs["dataloader"]["x_bound"]
    y_bound = cfgs["dataloader"]["y_bound"]
    feat_ratio = cfgs["backbone"]["bev_feat_ratio"]
    bev_width = int((x_bound[1] - x_bound[0]) / (x_bound[2]))
    bev_height = int((y_bound[1] - y_bound[0]) / (y_bound[2]))
    bev_feat_width = int(bev_width / feat_ratio)
    bev_feat_height = int(bev_height / feat_ratio)

    # 设定加载检测相关
    conf_threshold_detect = 0.15
    iou_threshold = 0.6
    obj_list = cfgs["detection"]["class_list"][1:]

    pm_matrix_np = np.load("data/matrix_nuscene.npy", allow_pickle=True)
    pm_matrix = torch.from_numpy(pm_matrix_np).cuda().unsqueeze(0).repeat(seq_num,1,1,1).unsqueeze(0).cuda().float()

    ego_motion_np = np.load("data/ego_motion_nuscene.npy", allow_pickle=True)
    seq_ego_motion = []
    for idx in range(seq_num):
        seq_ego_motion.append(torch.tensor(ego_motion_np[idx])) # 取前三个测试
    seq_ego_motion[-1] = torch.zeros(6)
    seq_ego_motion_tensor = torch.stack(seq_ego_motion, dim=0).unsqueeze(0).cuda().float()

    #---------------------------------------------------#
    #  网络模型
    #---------------------------------------------------#
    ultranet = VectorBEV(cfgs=cfgs, onnx_export=True, mask_path=mask_name).cuda()

    dict_old = torch.load(os.path.join(log_path,"model",model_name))
    if use_distribute:
        dict_new = deparallel_model(dict_old)
    else:
        dict_new = dict_old

    ultranet.load_state_dict(dict_new,strict=False)
    ultranet.eval()

    input_imgs = torch.zeros((1, seq_num, 6, 3, net_input_height, net_input_width)).cuda() + 1

    #---------------------------------------------------#
    #  MAC和参数量
    #---------------------------------------------------#
    # from thop import profile, clever_format
    # macs, params = profile(ultranet, inputs=(input_imgs,pm_matrix,seq_ego_motion_tensor,"valid"))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('MACs: {}'.format(macs))
    # print('Params: {}'.format(params))

    #---------------------------------------------------#
    #  GPU warm-up
    #---------------------------------------------------#
    t_all = 0
    for _ in range(10):
        torch.cuda.synchronize()
        t1 = time.time()

        ultranet(input_imgs, pm_matrix,seq_ego_motion_tensor,"deploy")

        torch.cuda.synchronize()
        t2 = time.time()
        t_all += t2 - t1

    print('Average latency (ms): {:.2f}'.format(t_all * 1000 / 10))
    print('Average FPS: {:.2f}'.format(10 / t_all))

    #---------------------------------------------------#
    # 单层推理时间
    #---------------------------------------------------#
    with torch.autograd.profiler.profile(use_cuda=True,profile_memory=True) as prof:
        ultranet(input_imgs, pm_matrix,seq_ego_motion_tensor, "deploy")
    print(prof.key_averages().table(sort_by="cuda_time_total") )
    # prof.export_chrome_trace('./profile.json')
    # print(prof.key_averages().table() )

    #---------------------------------------------------#
    #  这里进行profile分析
    #---------------------------------------------------#
    import torchprof
    with torchprof.Profile(ultranet, use_cuda=True) as prof:
        ultranet(input_imgs, pm_matrix, seq_ego_motion_tensor,"deploy")
    # print(prof.display(show_events=False)) # equivalent to `print(prof)` and `print(prof.display())`
    print(prof.display(show_events=True)) # 查看每个层中发生的低级操作


