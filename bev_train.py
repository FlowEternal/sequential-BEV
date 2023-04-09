"""
Function: VectorBEV Training Scripts
Author: Zhan Dong Xu
Date: 2021/11/11
"""

# 通用库导入
import os, yaml, time, shutil, cv2
import numpy as np
import prettytable as pt
import warnings
warnings.filterwarnings("ignore")

# 数据loader导入
from dataset.dataloader import BEVData

# 模型导入
from bev_model import VectorBEV
import torch.distributed
import torch.utils.data.dataloader

# VectorHeader Library
from bev_vector.devkit.evaluation.chamfer_distance import semantic_mask_chamfer_dist_cum
from bev_vector.devkit.evaluation.AP import instance_mask_AP
from bev_vector.devkit.evaluation.iou import get_batch_iou
from bev_vector.vector_header import print_val_info

# DetectHeader Library
from bev_det.centernet import bbox2result
from bev_det.rot_det_metric import eval_mAP
from bev_demo import sort_corners

class VectorBEVTrainer(object):
    def __init__(self, cfgs, cfg_path):
        self.cfgs = cfgs
        self.tag = self.cfgs["tag"]
        self.logs = self.cfgs["train"]["logs"]
        self.print_interval = self.cfgs["train"]["print_interval"]

        # 保存路径
        self.save_dir = os.path.join(self.logs, time.strftime('%d_%B_%Y_%H_%M_%S_%Z') + "_" + self.tag)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        # 配置文件
        self.cfg_path = cfg_path
        self.cfg_path_backup = os.path.join(self.save_dir,"config.yml")
        shutil.copy(self.cfg_path, self.cfg_path_backup)

        # 模型文件保存路径
        self.model_save_dir = os.path.join(self.save_dir,"model")
        if not os.path.exists(self.model_save_dir): os.makedirs(self.model_save_dir)

        # 并行训练初始化相关
        self.use_distribute = self.cfgs["train"]["use_distribute"]
        if self.use_distribute:
            torch.distributed.init_process_group(backend='nccl',
                                                 # init_method='tcp://localhost:23457',
                                                 init_method='tcp://localhost:13457',
                                                 rank=0,
                                                 world_size=1)

        #---------------------------------------------------#
        #  1.数据加载模块
        #---------------------------------------------------#
        self.batch_size_train = self.cfgs["train"]["batch_size_train"]
        self.num_worker_train = self.cfgs["train"]["num_worker_train"]

        self.batch_size_valid = self.cfgs["train"]["batch_size_valid"]
        self.num_worker_valid = self.cfgs["train"]["num_worker_valid"]

        self.net_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.net_input_height = self.cfgs["dataloader"]["network_input_height"]

        self.x_bound = self.cfgs["dataloader"]["x_bound"]
        self.y_bound = self.cfgs["dataloader"]["y_bound"]
        self.bev_feat_ratio = self.cfgs["backbone"]["bev_feat_ratio"]
        self.bev_width = int((self.x_bound[1] - self.x_bound[0]) / (self.x_bound[2]))
        self.bev_height = int((self.y_bound[1] - self.y_bound[0]) / (self.y_bound[2]))
        self.bev_feat_width = int(self.bev_width / self.bev_feat_ratio)
        self.bev_feat_height = int(self.bev_height / self.bev_feat_ratio)

        # training loader
        self.train_data = BEVData(cfgs=cfgs, mode="train")
        self.trainloader = torch.utils.data.dataloader.DataLoader(
                                                             self.train_data,
                                                             batch_size=self.batch_size_train,
                                                             num_workers=self.num_worker_train,
                                                             shuffle=True,
                                                             drop_last=False,
                                                             pin_memory=True,
                                                             collate_fn=self.train_data.collate_fn,
                                                             )

        # testing loader
        self.valid_data = BEVData(cfgs=cfgs, mode="val")
        self.validloader = torch.utils.data.dataloader.DataLoader(
                                                             self.valid_data,
                                                             batch_size=self.batch_size_valid,
                                                             num_workers=self.num_worker_valid,
                                                             shuffle=False,
                                                             drop_last=False,
                                                             pin_memory=True,
                                                             collate_fn=self.valid_data.collate_fn,
                                                             )

        #---------------------------------------------------#
        #  2.模型加载模块
        #---------------------------------------------------#
        self.ultrabev = VectorBEV(cfgs=cfgs).cuda()
        self.continue_train = self.cfgs["train"]["continue_train"]
        self.weight_file = self.cfgs["train"]["weight_file"]
        if self.continue_train:
            def deparallel_model(dict_param):
                ck_dict_new = dict()
                for key, value in dict_param.items():
                    temp_list = key.split(".")[1:]
                    new_key = ""
                    for tmp in temp_list:
                        new_key += tmp + "."
                    ck_dict_new[new_key[0:-1]] = value
                return ck_dict_new

            dict_old = torch.load(self.weight_file)
            if self.use_distribute:
                dict_new = deparallel_model(dict_old)
            else:
                dict_new = dict_old
            self.ultrabev.load_state_dict(dict_new,strict=False)

        # 并行训练开启与否
        if self.use_distribute:
            self.ultrabev = torch.nn.parallel.DistributedDataParallel(self.ultrabev, find_unused_parameters=True)

        #---------------------------------------------------#
        #  3.优化器 + loss权重
        #---------------------------------------------------#
        self.epoch = self.cfgs["train"]["epoch"]
        self.lr = self.cfgs["train"]["lr"]
        self.weight_decay = self.cfgs["train"]["weight_decay"]
        self.total_iters = len(self.trainloader) * self.epoch
        self.optimizer = torch.optim.Adam(self.ultrabev.parameters(), self.lr, weight_decay= self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_iters, eta_min=1e-8)

        # 训练时是否tuning每个分支
        self.tuning_single_header = self.cfgs["train"]["tuning_single_header"]
        self.tuning_interval = self.cfgs["train"]["tuning_interval"]

        # 每个分支权值
        self.vector_weight = self.cfgs["vector"]["vector_weight"]
        self.detection_weight = self.cfgs["detection"]["detection_weight"]

        #---------------------------------------------------#
        #  4.模型验证 -- 关键
        #---------------------------------------------------#
        # vector header
        self.class_list = self.cfgs["vector"]["class_list"]
        self.sampled_recalls_num = self.cfgs["vector"]["sampled_recalls_num"]
        self.thresholds = self.cfgs["vector"]["thresholds"]
        self.cd_threshold = self.cfgs["vector"]["cd_threshold"]
        self.resolution_x = self.cfgs["vector"]["resolution_x"]
        self.resolution_y = self.cfgs["vector"]["resolution_y"]
        self.thickness = self.cfgs["vector"]["thickness"]
        self.max_instance_num = self.cfgs["vector"]["max_instance_num"]
        self.SAMPLED_RECALLS = torch.linspace(0.1, 1, self.sampled_recalls_num)
        self.CHANNEL_NAMES = self.class_list
        self.max_channel = len(self.CHANNEL_NAMES)

        # detector header
        self.num_class_detect = cfgs["detection"]["num_classes"]
        self.class_list = cfgs["detection"]["class_list"][1:]

        self.root_dir = self.cfgs["dataloader"]["data_list"].replace("/list","")
        self.gt_label_path = os.path.join(self.root_dir,"eval_gt_det")
        self.pred_label_path = os.path.join(self.root_dir,"eval_pred_det")

        if not os.path.exists(self.gt_label_path): os.makedirs(self.gt_label_path)
        if not os.path.exists(self.pred_label_path): os.makedirs(self.pred_label_path)

        def remove_recursive(folder):
            list_item = os.listdir(folder)
            if len(list_item) == 0:
                return
            else:
                for tmp in list_item:
                    os.remove(os.path.join(folder, tmp))
        remove_recursive(self.gt_label_path)
        remove_recursive(self.pred_label_path)

        # 产生真值标签
        val_dataset_info = self.valid_data.image_annot_path_pairs
        for item in val_dataset_info:
            src_det_path = item["annot_path_detect"]
            src_label = open(src_det_path, "r").readlines()
            folder_id = os.path.basename(os.path.dirname(os.path.dirname(src_det_path)))
            dst_det_path = os.path.join(self.gt_label_path, str(folder_id) + ".txt")
            dst_writer = open(dst_det_path, "w")
            for one_line in src_label:
                tmp_info = one_line.strip("\n").split(",")
                class_name = tmp_info[0]
                x1 = float(tmp_info[1])
                y1 = float(tmp_info[2])

                x2 = float(tmp_info[3])
                y2 = float(tmp_info[4])

                x3 = float(tmp_info[5])
                y3 = float(tmp_info[6])

                x4 = float(tmp_info[7])
                y4 = float(tmp_info[8])

                # 逆时针转90度后
                x1_ = y1
                y1_ = self.bev_height - x1
                flag_in_range1_ = (0 <= x1_ < self.bev_width - 1) and (0 <= y1_ < self.bev_height - 1)

                x2_ = y2
                y2_ = self.bev_height - x2
                flag_in_range2_ = (0 <= x2_ < self.bev_width - 1) and (0 <= y2_ < self.bev_height - 1)

                x3_ = y3
                y3_ = self.bev_height - x3
                flag_in_range3_ = (0 <= x3_ < self.bev_width - 1) and (0 <= y3_ < self.bev_height - 1)

                x4_ = y4
                y4_ = self.bev_height - x4
                flag_in_range4_ = (0 <= x4_ < self.bev_width - 1) and (0 <= y4_ < self.bev_height - 1)
                pts = np.array([x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_]).reshape(4,2)
                flag_merge = flag_in_range1_ and flag_in_range2_ and flag_in_range3_ and flag_in_range4_
                if not flag_merge:
                    continue

                det_info = '{},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                    class_name, pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1])
                dst_writer.writelines(det_info)
            dst_writer.close()
            # shutil.copy(src_det_path, dst_det_path)

    def cal_total_loss(self, loss_dict, epoch):
        loss_total = 0.0

        vector_weight = self.vector_weight
        detection_weight = self.detection_weight

        if self.tuning_single_header:
            if self.tuning_interval <= epoch %  (3 * self.tuning_interval) < 2 * self.tuning_interval:
                # train vector
                detection_weight = 0.0
            if 2 * self.tuning_interval <= epoch %  (3 * self.tuning_interval) < 3 * self.tuning_interval:
                # train detector
                vector_weight = 0.0

        # Vector
        loss_total += (loss_dict["loss_center_map"]  +
                       loss_dict["loss_offset"]  +
                       loss_dict["loss_cluster"] +
                       loss_dict["loss_cls"] +
                       loss_dict["loss_direct"]) * vector_weight

        # 检测
        loss_total += (loss_dict["loss_detect_center_heatmap"]  +
                       loss_dict["loss_detect_wh"]  +
                       loss_dict["loss_detect_offset"] +
                       loss_dict["loss_detect_rotation"]) * detection_weight

        return loss_total

    def print_loss_info(self, loss_dict, epoch, batch_idx, mode="train"):
        if mode == "train":
            print("TRAIN Epoch [%i|%i] Iter [%i|%i] Lr %.5f" % (epoch , self.epoch,
                                                                batch_idx , len(self.trainloader),
                                                                self.optimizer.param_groups[0]["lr"]))
        else:
            print("VALID Epoch [%i|%i] Iter [%i|%i] Lr %.5f" % (epoch , self.epoch,
                                                                batch_idx , len(self.validloader),
                                                                self.optimizer.param_groups[0]["lr"]))

        tb = pt.PrettyTable()
        row_list = list()
        key_list = list()
        for key, value in loss_dict.items():
            value_str = float("%.3f" %value.item())
            row_list.append(value_str)
            key_list.append(key)

        tb.field_names = key_list
        tb.add_row(row_list)
        print(tb)
        print()

    @staticmethod
    def to_gpu(batch_data):
        batch_data["image"] = batch_data["image"].cuda().float()
        batch_data["pm_matrix"] = batch_data["pm_matrix"].cuda().float()
        batch_data["ego_motion"] = batch_data["ego_motion"].cuda().float()

        # vector
        batch_data["gt_binary"]=batch_data["gt_binary"].cuda().float()
        batch_data["gt_offsetmap"]=batch_data["gt_offsetmap"].cuda().float()
        batch_data["gt_instancemap"]=batch_data["gt_instancemap"].cuda().float()
        batch_data["gt_classmap"]=batch_data["gt_classmap"].cuda().float()
        batch_data["gt_point_direction"]=batch_data["gt_point_direction"].cuda().float()

        # detect
        for i in range(len(batch_data["gt_det_bboxes"])):
            batch_data["gt_det_bboxes"][i] = torch.tensor(batch_data["gt_det_bboxes"][i]).cuda().float()
            batch_data["gt_det_labels"][i] = torch.tensor(batch_data["gt_det_labels"][i]).cuda().long()

        return batch_data

    def train_one_epoch(self, epoch):
        self.ultrabev.train()
        for iter_idx, batch_data in enumerate(self.trainloader):

            # forward pass
            batch_data = self.to_gpu(batch_data)
            inputs = batch_data["image"]
            pm_matrix = batch_data["pm_matrix"]
            ego_motion = batch_data["ego_motion"]
            outputs = self.ultrabev(inputs, pm_matrix, ego_motion)
            if self.use_distribute:
                loss_dict = self.ultrabev.module.cal_loss(outputs, batch_data)
            else:
                loss_dict = self.ultrabev.cal_loss(outputs, batch_data)

            loss_total = self.cal_total_loss(loss_dict, epoch)
            loss_dict.update({"total_loss": loss_total})

            # 打印结果
            if iter_idx % self.print_interval ==0:
                self.print_loss_info(loss_dict, epoch, iter_idx,mode="train")

            self.optimizer.zero_grad()
            loss_total.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            self.optimizer.step()

            # 调整学习率
            self.scheduler.step()

        return

    def valid(self, epoch):
        self.ultrabev.eval()

        total_CD1 = torch.zeros(self.max_channel)
        total_CD2 = torch.zeros(self.max_channel)
        total_CD_num1 = torch.zeros(self.max_channel)
        total_CD_num2 = torch.zeros(self.max_channel)
        total_intersect = torch.zeros(self.max_channel)
        total_union = torch.zeros(self.max_channel)
        AP_matrix = torch.zeros((self.max_channel, len(self.thresholds)))
        AP_count_matrix = torch.zeros((self.max_channel, len(self.thresholds)))

        for iter_idx , batch_data in enumerate(self.validloader):
            # forward pass
            batch_data = self.to_gpu(batch_data)
            inputs = batch_data["image"]
            pm_matrix = batch_data["pm_matrix"]
            ego_motion = batch_data["ego_motion"]
            outputs = self.ultrabev(inputs, pm_matrix, ego_motion)
            batch_size_tmp = inputs.shape[0]
            if self.use_distribute:
                loss_dict = self.ultrabev.module.cal_loss(outputs, batch_data)
            else:
                loss_dict = self.ultrabev.cal_loss(outputs, batch_data)

            loss_total = self.cal_total_loss(loss_dict,epoch)
            loss_dict.update({"total_loss": loss_total})

            # 打印结果
            self.print_loss_info(loss_dict, epoch, iter_idx, mode="valid")

            #---------------------------------------------------#
            #  Vector Evaluation
            #---------------------------------------------------#
            out_confidence = outputs["confidence"]
            out_offset = outputs["offset"]
            out_instance = outputs["instance"]
            out_direct = outputs["direct"]
            out_cls = outputs["cls"]
            output_ = [out_confidence, out_offset, out_instance,out_direct, out_cls]

            if self.use_distribute:
                decoded_vector_list = self.ultrabev.module.vector_header.decode_result_vector(output_)
            else:
                decoded_vector_list = self.ultrabev.vector_header.decode_result_vector(output_)

            # ---------------------------------------------------#
            #  generate prediction map and confidence list
            # ---------------------------------------------------#
            masks_batches = list()
            confidence_level_list = list()
            for one_sampel_vector_list in decoded_vector_list:
                confidence_level_ = [-1]
                decoded_vector_dict = {"0": [], "1": [], "2": []}
                decoded_vector_confidence = {"0": [], "1": [], "2": []}

                for one_vector in one_sampel_vector_list:
                    type_ = str(one_vector["type_"])
                    decoded_vector_dict[type_].append(one_vector["pts"])
                    decoded_vector_confidence[type_].append(one_vector["confidence_level"])

                # convert to pred map
                idx = 1
                thickness = self.thickness
                masks = []
                for key, value_container in decoded_vector_dict.items():
                    map_mask = np.zeros((self.bev_height, self.bev_width), np.uint8)
                    confidence_container = decoded_vector_confidence[key]
                    for one_line, confidence in zip(value_container, confidence_container):

                        if self.use_distribute:
                            map_mask, idx = self.ultrabev.module.vector_header.mask_for_lines(one_line, map_mask, idx, thickness)  #

                        else:
                            map_mask, idx = self.ultrabev.vector_header.mask_for_lines(one_line, map_mask, idx, thickness)  #
                        confidence_level_.append(confidence)
                    masks.append(map_mask)

                masks_one_sample = torch.tensor(np.stack(masks))
                masks_batches.append(masks_one_sample)

                confidence_level = torch.tensor(confidence_level_ + [-1] * (self.max_instance_num - len(confidence_level_)))
                confidence_level_list.append(confidence_level)

            pred_map = torch.stack(masks_batches)
            confidence_level = torch.stack(confidence_level_list)

            # ---------------------------------------------------#
            #  generate gt map list
            # ---------------------------------------------------#
            gt_map = []
            vector_list_tmp = batch_data["vector_metas"]
            for vector_label in vector_list_tmp:

                # convert to gt map
                idx = 1
                thickness = self.thickness
                masks = []
                for key, value_container in vector_label.items():
                    map_mask = np.zeros((self.bev_height, self.bev_width), np.uint8)
                    for one_line in value_container:
                        if self.use_distribute:
                            map_mask, idx = self.ultrabev.module.vector_header.mask_for_lines(one_line, map_mask, idx, thickness)  #

                        else:
                            map_mask, idx = self.ultrabev.vector_header.mask_for_lines(one_line, map_mask, idx, thickness)  #

                    masks.append(map_mask)

                masks_one_sample = torch.tensor(np.stack(masks))
                gt_map.append(masks_one_sample)
            gt_map = torch.stack(gt_map)

            # gt map && confidences && predict map calculation
            # iou
            intersect, union = get_batch_iou(pred_map, gt_map)
            CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(pred_map, gt_map, self.resolution_x, self.resolution_y,
                                                                  threshold=self.cd_threshold)

            instance_mask_AP(AP_matrix, AP_count_matrix, pred_map, gt_map, self.resolution_x, self.resolution_y,
                             confidence_level, self.thresholds, sampled_recalls=self.SAMPLED_RECALLS)

            total_intersect += intersect
            total_union += union
            total_CD1 += CD1
            total_CD2 += CD2
            total_CD_num1 += num1
            total_CD_num2 += num2


            #---------------------------------------------------#
            #  检测部分metric
            #---------------------------------------------------#
            img_name_list = batch_data["img_name_list"]

            for idx in range(batch_size_tmp):
                tmp_img_path = img_name_list[idx]
                outs_detect = ([outputs["detection"][0][0][idx].unsqueeze(0)],
                               [outputs["detection"][1][0][idx].unsqueeze(0)],
                               [outputs["detection"][2][0][idx].unsqueeze(0)],
                               [outputs["detection"][3][0][idx].unsqueeze(0)],
                               )
                img_metas = batch_data["img_metas"][idx]
                if self.use_distribute:
                    results_list = self.ultrabev.module.center_head.get_bboxes(*outs_detect, img_metas, rescale=True)

                else:
                    results_list = self.ultrabev.center_head.get_bboxes(*outs_detect, img_metas, rescale=True)

                bbox_result = [
                    bbox2result(det_bboxes, det_labels, self.num_class_detect)
                    for det_bboxes, det_labels in results_list
                ]

                for idx_ in range(len(bbox_result)):
                    target_pred_txt_path = os.path.join(self.pred_label_path, tmp_img_path + ".txt")
                    writer_this = open(target_pred_txt_path, "w")

                    bbox_result = bbox_result[idx_]
                    for cls_id, one_class_result in enumerate(bbox_result):
                        if one_class_result.shape[0] == 0:
                            continue
                        else:
                            this_cls_num = one_class_result.shape[0]
                            for i in range(this_cls_num):
                                x1 = one_class_result[i, 0]
                                y1 = one_class_result[i, 1]
                                x2 = one_class_result[i, 2]
                                y2 = one_class_result[i, 3]
                                rot_angel = one_class_result[i, 4]
                                score = one_class_result[i, 5]

                                x = (x1 + x2) / 2.0
                                y = (y1 + y2) / 2.0
                                w = max((x2 - x1), 0)
                                h = max((y2 - y1), 0)
                                pts = cv2.boxPoints(((x, y), (w, h), rot_angel * 180 / 3.141592)).reshape(1, 8)
                                pts = sort_corners(pts)[0]

                                det_info = '{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                                    self.class_list[cls_id], score,
                                    pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7])

                                writer_this.writelines(det_info)

                    writer_this.close()

        # ---------------------------------------------------#
        #  Vector部分metric summary
        # ---------------------------------------------------#
        CD_pred = total_CD1 / total_CD_num1
        CD_label = total_CD2 / total_CD_num2
        CD = (total_CD1 + total_CD2) / (total_CD_num1 + total_CD_num2)
        CD_pred[CD_pred > self.cd_threshold] = self.cd_threshold
        CD_label[CD_label > self.cd_threshold] = self.cd_threshold
        CD[CD > self.cd_threshold] = self.cd_threshold

        dict_info = {
            'iou': total_intersect / total_union,
            'CD_pred': CD_pred,
            'CD_label': CD_label,
            'CD': CD,
            'Average_precision': AP_matrix / AP_count_matrix,
        }

        # evaluate
        print_val_info(dict_info, self.CHANNEL_NAMES)

        # ---------------------------------------------------#
        #  检测部分metric summary
        # ---------------------------------------------------#
        mAP = eval_mAP(ROOT_DIR=self.root_dir,
                       VALID_DETECT_GROUND_TRUTH=self.gt_label_path,
                       VALID_DETECT_RESULT=self.pred_label_path,
                       use_07_metric=False)
        print("mAP is: %.2f" % mAP)

        #---------------------------------------------------#
        # save model
        #---------------------------------------------------#
        torch.save(self.ultrabev.state_dict(), os.path.join(self.model_save_dir,"epoch_%i.pth" % epoch))
        return

def main(cfg_path):
    cfgs = yaml.safe_load(open(cfg_path))
    trainer = VectorBEVTrainer(cfgs, cfg_path)
    epoch_all = cfgs["train"]["epoch"]
    for epoch in range(epoch_all):
        trainer.train_one_epoch(epoch)
        print("=========================== VALIDATION %i ===========================" %epoch)
        trainer.valid(epoch)
    print("============== finish training ==============")
    return


def gpu_set(gpu_begin, gpu_number):
    gpu_id_str = ""
    for i in range(gpu_begin, gpu_number + gpu_begin):
        if i != gpu_begin + gpu_number - 1:
            gpu_id_str = gpu_id_str + str(i) + ","
        else:
            gpu_id_str = gpu_id_str + str(i)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    gpu_set(0, 4)

    # cfg_path = "cfgs/ultrabev_stn_seq1_pretrain.yml"
    cfg_path = "cfgs/ultrabev_stn_seq1_finetune.yml"

    # cfg_path = "cfgs/ultrabev_stn_seq2_pretrain.yml"
    # cfg_path = "cfgs/ultrabev_stn_seq2_finetune.yml"

    # cfg_path = "cfgs/ultrabev_stn_seq3_pretrain.yml"
    # cfg_path = "cfgs/ultrabev_stn_seq3_finetune.yml"

    main(cfg_path)

    # 杀死所有进程号从2开始的进程
    # fuser -v /dev/nvidia* | grep 2* | xargs kill -9
