"""
Function: VectorBEV Model Demo
Author: Zhan Dong Xu
Date: 2021/11/17
"""

import yaml, os, time
from queue import Queue
import numpy as np
import cv2

import warnings
warnings.filterwarnings("ignore")

from shapely.geometry import Polygon
import torch.utils.data.dataloader

# main model
from bev_model import VectorBEV
from bev_det.centernet import  bbox2result


def imagenet_normalize(img):
    """Normalize image.

    :param img: img that need to normalize
    :type img: RGB mode ndarray
    :return: normalized image
    :rtype: numpy.ndarray
    """
    pixel_value_range = np.array([255, 255, 255])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / pixel_value_range
    img = img - mean
    img = img / std
    return img


def deparallel_model(dict_param):
    ck_dict_new = dict()
    for key, value in dict_param.items():
        temp_list = key.split(".")[1:]
        new_key = ""
        for tmp in temp_list:
            new_key += tmp + "."
        ck_dict_new[new_key[0:-1]] = value
    return ck_dict_new


def decode(imgs, masks, org_size, vis_color_id):
    seg_predictions = torch.argmax(masks, dim=1).detach().cpu().numpy()

    batch_size = len(imgs)
    visual_imgs = list()
    for batch_idx in range(batch_size):
        seg_prediction = seg_predictions[batch_idx]
        im_vis = imgs[batch_idx]

        # vis
        vis_seg = np.zeros([seg_prediction.shape[0], seg_prediction.shape[1], 3], dtype=np.uint8)
        for cls_id, color in vis_color_id.items():
            vis_seg[seg_prediction == cls_id] = color
        vis_seg = cv2.resize(vis_seg, org_size, cv2.INTER_NEAREST)
        im_vis = cv2.addWeighted(im_vis, 0.8, vis_seg, 0.5, 0.0)
        visual_imgs.append(im_vis)

    return visual_imgs


def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j*2] = corners[idx*2]
            sorted[i, j*2+1] = corners[idx*2+1]
    return sorted


if __name__ == '__main__':
    # torch.cuda.set_device("cuda:2")
    deploy = False
    display = True
    compare_onnxruntime = False
    SKIP_INDEX_one = [40,81,122,163,203,242,283,323,363,403]
    SKIP_INDEX_two = [39,80,121,162,202,241,282,322,362,402]
    VIDEO_SQUEEZE_RATIO = 2

    #---------------------------------------------------#
    #  选择输入模型
    #---------------------------------------------------#
    ############# seq = 1 #############
    # transformer encoder decoder的效果不是特别好
    # log_path = "logs/transformer_seq1_pretrain"
    # model_name = "epoch_39.pth"

    # stn pretrain
    # log_path = "logs/stn_seq1_pretrain"
    # model_name = "epoch_39.pth"

    # stn finetune
    log_path = "logs/stn_seq1_finetune"
    model_name = "epoch_11.pth"

    ############# seq = 2 #############
    # stn pretrain
    # log_path = "logs/stn_seq2_pretrain"
    # model_name = "epoch_39.pth"

    # stn finetune
    # log_path = "logs/stn_seq2_finetune"
    # model_name = "epoch_11.pth"

    ############# seq = 3 #############
    # stn pretrain
    # log_path = "logs/stn_seq3_pretrain"
    # model_name = "epoch_39.pth"

    # stn finetune
    # log_path = "logs/stn_seq3_finetune"
    # model_name = "epoch_11.pth"

    # 输入数据
    video_root = "data/video_nuscene"
    mask_name = "data/mask_nuscene.npy"

    # 检测相关
    conf_threshold_detect = 0.30

    #---------------------------------------------------#
    #  参数导出
    #---------------------------------------------------#
    # 导出参数
    cfg_path = os.path.join(log_path, "config.yml")
    cfgs = yaml.safe_load(open(cfg_path))
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    cam_list = cfgs["dataloader"]["camera_list"]

    # 时序相关参数 -- sequence related parameter
    seq_num = cfgs["dataloader"]["seq_num"]
    q_input_images = Queue(maxsize=seq_num)
    q_ego_motion = Queue(maxsize=seq_num)

    # for transfomer encoder - decoder
    layer_id = cfgs["backbone"]["layer_id"]
    use_transformer_decoder = cfgs["backbone"]["use_transformer_decoder"]
    feat_width = int(net_input_width / pow(2, layer_id + 1))
    feat_height = int(net_input_height / pow(2, layer_id + 1))

    net_input_size = (net_input_width, net_input_height)
    use_distribute = cfgs["train"]["use_distribute"]
    x_bound = cfgs["dataloader"]["x_bound"]
    y_bound = cfgs["dataloader"]["y_bound"]
    feat_ratio = cfgs["backbone"]["bev_feat_ratio"]
    bev_width = int((x_bound[1] - x_bound[0]) / (x_bound[2]))
    bev_height = int((y_bound[1] - y_bound[0]) / (y_bound[2]))
    bev_feat_width = int(bev_width / feat_ratio)
    bev_feat_height = int(bev_height / feat_ratio)
    obj_list = cfgs["detection"]["class_list"][1:]

    # 变换矩阵
    pm_matrix_np = np.load("data/matrix_nuscene.npy", allow_pickle=True)
    # for tmp_mat in pm_matrix_np:
    #     print(np.reshape(tmp_mat,-1))
    pm_matrix = torch.from_numpy(pm_matrix_np).cuda().unsqueeze(0).repeat(seq_num,1,1,1).unsqueeze(0).cuda().float()

    # ego motion 矩阵
    ego_motion_np = np.load("data/ego_motion_nuscene.npy", allow_pickle=True)

    #---------------------------------------------------#
    #  网络模型
    #---------------------------------------------------#
    if deploy:
        ultranet = VectorBEV(cfgs=cfgs, onnx_export=True, mask_path=mask_name).cuda()
    else:
        ultranet = VectorBEV(cfgs=cfgs, onnx_export=False,mask_path=mask_name).cuda()

    dict_old = torch.load(os.path.join(log_path,"model",model_name))
    if use_distribute:
        dict_new = deparallel_model(dict_old)
    else:
        dict_new = dict_old

    ultranet.load_state_dict(dict_new,strict=False)
    ultranet.eval()

    #---------------------------------------------------#
    #  导出模型onnx
    #---------------------------------------------------#
    if deploy:
        import torch.onnx
        output_list = ["vec_confidence", "vec_offset", "vec_instance", "vec_direct", "vec_cls",
                       "topk_scores", "topk_inds", "wh_pred", "offset_pred", "rotation_pred"]
        dummy_input = torch.randn([1,seq_num, 6, 3, net_input_height, net_input_width]).to("cuda:0")
        pm_matrix = pm_matrix.to("cuda:0")
        dummy_ego_motion_matrix = torch.randn(1, seq_num, 6).to("cuda:0")
        torch.onnx.export(
            ultranet,(dummy_input, pm_matrix, dummy_ego_motion_matrix,"deploy"),
            "vectorbev_stn_seq%i.onnx" %seq_num,
            export_params=True,
            input_names=["input", "calibration","ego_motion","mode"],
            output_names=output_list,
            opset_version=11,
            verbose=False,
        )

        exit()

    #---------------------------------------------------#
    #  测试模型
    #---------------------------------------------------#
    video_list = [os.path.join(video_root, cam + ".avi") for cam in cam_list]
    vid_list = []
    for video_path in video_list:
        vid = cv2.VideoCapture(video_path)
        vid_list.append(vid)

    log_path_label = os.path.basename(log_path)
    video_output = video_root + "_%s_%s.mp4" % (log_path_label, model_name.split(".")[0])
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(video_output, codec, 20, ( int((320*3+240)/VIDEO_SQUEEZE_RATIO),int(480/VIDEO_SQUEEZE_RATIO)))
    img_list = None

    if display:
        cv2.namedWindow("visual",cv2.WINDOW_FREERATIO)

    counter = -1
    while True:
        input_img_list = list()
        vis_img_list = list()
        for vid in vid_list:
            _, input_img = vid.read()
            if input_img is None:
                exit()

            #---------------------------------------------------#
            #  preprocess
            #---------------------------------------------------#
            img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, net_input_size)
            vis_img_list.append(img.copy())
            img = img.astype(np.float32)
            img = imagenet_normalize(img)
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img).cuda().float()
            input_img_list.append(img)

        counter += 1
        if counter in SKIP_INDEX_one or counter in SKIP_INDEX_two:
            q_input_images.queue.clear()
            q_ego_motion.queue.clear()
            continue

        input_imgs = torch.stack(input_img_list,dim=0).cuda()

        # 队列操作
        if not q_input_images.full():
            q_input_images.put(input_imgs)
            q_ego_motion.put(ego_motion_np[counter])

            if not q_input_images.full():
                continue
        else:
            q_input_images.get(0)
            q_ego_motion.get(0)
            q_input_images.put(input_imgs)
            q_ego_motion.put(ego_motion_np[counter])

        seq_input_tensor = []
        seq_ego_motion = []
        for idx in range(seq_num):
            seq_input_tensor.append(q_input_images.queue[idx])
            seq_ego_motion.append( torch.tensor(q_ego_motion.queue[idx]))

        seq_input_tensor = torch.stack(seq_input_tensor, dim=0).unsqueeze(0)
        seq_ego_motion[-1] = torch.zeros(6)
        seq_ego_motion_tensor = torch.stack(seq_ego_motion, dim=0).unsqueeze(0).cuda().float()

        #---------------------------------------------------#
        #  inference
        #---------------------------------------------------#
        tic = time.time()
        outputs = ultranet(seq_input_tensor,pm_matrix,seq_ego_motion_tensor)
        print("inference time is %i" %(1000*(time.time() - tic)))


        if compare_onnxruntime:
            import torch.nn.functional as F
            import onnxruntime as rt

            # pytorch输出
            hmax = F.max_pool2d(outputs["detection"][0][0], 3, stride=1, padding=1)
            heat_point = (hmax == outputs["detection"][0][0]).float() * outputs["detection"][0][0]
            topk_scores_torch, topk_inds_torch = torch.topk(heat_point.view(1, -1), 100)  # 这里取top 100

            ##
            confidence_torch = outputs["confidence"].detach().cpu().numpy()
            offset_torch = outputs["offset"].detach().cpu().numpy()
            instance_torch = outputs["instance"].detach().cpu().numpy()
            direct_torch = outputs["direct"].detach().cpu().numpy()
            cls_torch = outputs["cls"].detach().cpu().numpy()

            topk_scores_torch = topk_scores_torch.detach().cpu().numpy()
            topk_inds_torch = topk_inds_torch.detach().cpu().numpy()
            wh_pred_torch = outputs["detection"][1][0].detach().cpu().numpy()
            offset_pred_torch = outputs["detection"][2][0].detach().cpu().numpy()
            rotation_pred_torch = outputs["detection"][3][0].detach().cpu().numpy()

            # onnxruntime 输出
            sess = rt.InferenceSession("vectorbev_stn_seq1.onnx", providers=['CUDAExecutionProvider'])
            provider = sess.get_providers()
            input_tensor_name = sess.get_inputs()[0].name
            input_pm_matrix = sess.get_inputs()[1].name
            vec_0 = sess.get_outputs()[0].name
            vec_1 = sess.get_outputs()[1].name
            vec_2 = sess.get_outputs()[2].name
            vec_3 = sess.get_outputs()[3].name
            vec_4 = sess.get_outputs()[4].name
            topk_scores = sess.get_outputs()[5].name
            topk_inds = sess.get_outputs()[6].name
            wh_pred = sess.get_outputs()[7].name
            offset_pred = sess.get_outputs()[8].name
            rotation_pred = sess.get_outputs()[9].name
            output_ = sess.run([vec_0, vec_1, vec_2, vec_3, vec_4,
                               topk_scores, topk_inds, wh_pred, offset_pred, rotation_pred],
                              {input_tensor_name: seq_input_tensor.detach().cpu().numpy(),
                               input_pm_matrix: pm_matrix.detach().cpu().numpy(),
                               })

            ##
            confidence_onnx = output_[0]
            offset_onnx = output_[1]
            instance_onnx = output_[2]
            direct_onnx = output_[3]
            cls_onnx = output_[4]
            topk_scores_onnx = output_[5]
            topk_inds_onnx = output_[6]
            wh_pred_onnx = output_[7]
            offset_pred_onnx = output_[8]
            rotation_pred_onnx = output_[9]

            print("onnxruntime output")
            print(topk_scores_onnx[0,0:3])
            print(topk_inds_onnx[0,0:3])
            print(wh_pred_onnx[0,0,0,0:3])
            print(offset_pred_onnx[0,0,0,0:3])
            print(rotation_pred_onnx[0,0,0,0:3])

            print("pytorch original output")
            print(topk_scores_torch[0,0:3])
            print(topk_inds_torch[0,0:3])
            print(wh_pred_torch[0,0,0,0:3])
            print(offset_pred_torch[0,0,0,0:3])
            print(rotation_pred_torch[0,0,0,0:3])
            exit()

        #---------------------------------------------------#
        #  decoder
        #---------------------------------------------------#
        org_size = (bev_width, bev_height)
        imgs = [np.zeros([bev_height, bev_width, 3]).astype(np.uint8)]

        # vector display
        out_confidence = outputs["confidence"]
        out_offset = outputs["offset"]
        out_instance = outputs["instance"]
        out_direct = outputs["direct"]
        out_cls = outputs["cls"]
        outs = [out_confidence, out_offset, out_instance, out_direct, out_cls]
        decoded_vector_list = ultranet.vector_header.decode_result_vector(outs)
        imgs = ultranet.vector_header.display(imgs, decoded_vector_list)

        # detection display
        target_size = (bev_height, bev_width)
        num_classes = cfgs["detection"]["num_classes"]
        detect_outs = outputs["detection"]
        img_meta = dict()
        img_meta["ori_shape"] = (bev_height, bev_width, 3)
        img_meta["img_shape"] = (bev_height, bev_width, 3)
        img_meta["pad_shape"] = (bev_height, bev_width, 3)
        img_meta["scale_factor"] = [1., 1., 1., 1.]
        img_meta["flip"] = False
        img_meta["flip_direction"] = None
        img_meta["border"] = [0, bev_height, 0, bev_width]
        img_meta["batch_input_shape"] = (bev_height, bev_width)
        img_metas = [img_meta]
        results_list = ultranet.center_head.get_bboxes(*detect_outs, img_metas, rescale=True)
        bbox_results = [bbox2result(det_bboxes, det_labels, num_classes)
                        for det_bboxes, det_labels in results_list][0]

        # filter bboxes
        bbox_result = bbox_results
        for cls_id, one_class_result in enumerate(bbox_result):
            if one_class_result.shape[0] == 0:
                continue
            else:
                this_cls_num = one_class_result.shape[0]
                for i in range(this_cls_num):
                    x1 = one_class_result[i,0]
                    y1 = one_class_result[i,1]
                    x2 = one_class_result[i,2]
                    y2 = one_class_result[i,3]
                    rot_angel = one_class_result[i,4]
                    score = one_class_result[i,5]

                    x = (x1 + x2) /2.0
                    y = (y1 + y2) /2.0
                    w = max((x2 - x1),0)
                    h = max((y2 - y1),0)
                    pts = cv2.boxPoints(((x,y),(w,h), -rot_angel * 180 / 3.141592)).reshape(1,8)
                    pts = sort_corners(pts)[0]
                    det_info = '{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                        obj_list[cls_id],score,
                        pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7])

                    if score > conf_threshold_detect:
                        pt1 = int(pts[0]), int(pts[1])
                        pt2 = int(pts[2]), int(pts[3])
                        pt3 = int(pts[4]), int(pts[5])
                        pt4 = int(pts[6]), int(pts[7])
                        # cv2.line(imgs[0], pt1, pt2, (0,255,0),2)
                        # cv2.line(imgs[0], pt2, pt3, (255,0,0),2)
                        # cv2.line(imgs[0], pt3, pt4, (0,0,255),2)
                        # cv2.line(imgs[0], pt4, pt1, (255,255,0),2)

                        poly_rect = Polygon([pt1,pt2,pt3,pt4])
                        if poly_rect.area > 400:
                            cv2.fillPoly(imgs[0],[pts.astype(np.int32).reshape(4,2)],(0,0,255))

        print("total process time is %i" %(1000*(time.time() - tic)))

        #---------------------------------------------------#
        #  Merge显示结果
        #---------------------------------------------------#
        img_draw = np.zeros([ 480, 320*3 + 240 ,3]).astype(np.uint8)
        height = 240
        width = 320
        DELTA_X = 25
        DELTA_Y = 40
        txt_org = (0 + DELTA_X, 0 + DELTA_Y)
        font_scale = 2
        thickness = 4
        color = (0, 255, 255)
        M1 = 7
        M2 = 14

        # front
        img_front = cv2.putText(vis_img_list[0],"front",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[0:height,width: 2*width,:] = cv2.resize(img_front,(width,height))

        # front left
        img_front_left = cv2.putText(vis_img_list[1],"front left",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[0:height,0:width,:] = cv2.resize(img_front_left,(width,height))

        # front right
        img_front_right = cv2.putText(vis_img_list[2],"front right",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[0:height,2*width: 3*width,:] = cv2.resize(img_front_right,(width,height))

        # back
        img_back = cv2.putText(vis_img_list[3],"back",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[height:2*height,width: 2*width,:] = cv2.resize(img_back,(width,height))

        # back left
        img_back_left = cv2.putText(vis_img_list[4],"back left",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[height:2*height,0: width,:] = cv2.resize(img_back_left,(width,height))

        # back right
        img_back_right = cv2.putText(vis_img_list[5],"back right",txt_org,cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness)
        img_draw[height:2*height,2*width: 3*width,:] = cv2.resize(img_back_right,(width,height))

        # bev
        bev_vis = imgs[0]
        bev_vis = cv2.resize(bev_vis, (2 * height, height))

        bev_vis = cv2.flip(bev_vis, 1)
        bev_vis = cv2.transpose(bev_vis)
        cv2.rectangle(bev_vis, (int(height/2) - M1, int(2 *height/2) - M2),
                      (int(height/2) + M1, int(2 *height/2) + M2), (0, 255, 255), -1)

        img_draw[0:2*height,3*width:,:] = bev_vis

        # ---------------------------------------------------#
        #  保存显示
        # ---------------------------------------------------#
        if display:
            cv2.imshow('visual', img_draw)
            cv2.waitKey(1)

        img_draw = cv2.resize(img_draw,( int(img_draw.shape[1]/2) , int(img_draw.shape[0]/2) ))
        video_writer.write(img_draw)


