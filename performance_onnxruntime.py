import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import time

if __name__ == '__main__':
    #---------------------------------------------------#
    #  参数设定
    #---------------------------------------------------#
    # onnx_save_dir = "vectorbev_stn_seq1.onnx"
    # seq_num = 1

    onnx_save_dir = "vectorbev_stn_seq2.onnx"
    seq_num = 2

    # onnx_save_dir = "vectorbev_stn_seq3.onnx"
    # seq_num = 3

    pm_matrix_dir = "data/matrix_nuscene.npy"

    tensor = torch.randn(1,seq_num,6,3,480,640).cuda()
    pm_matrix = torch.from_numpy(np.load(pm_matrix_dir, allow_pickle=True)).cuda().unsqueeze(0).repeat(seq_num,1,1,1).unsqueeze(0).cuda().float()
    dummy_ego_motion_matrix = torch.randn(1, seq_num, 6).to("cuda:0")

    # onnxruntime input
    onnxruntime_input_tensor = tensor.detach().cpu().numpy()
    onnxruntime_pm_tensor = pm_matrix.detach().cpu().numpy()
    dummy_ego_motion_matrix_tensor = dummy_ego_motion_matrix.detach().cpu().numpy()

    #---------------------------------------------------#
    #  onnxruntime推理
    #---------------------------------------------------#
    import onnxruntime as rt
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = rt.InferenceSession(onnx_save_dir,providers=EP_list)
    provider = sess.get_providers()
    print(provider[0])

    input_tensor_name = sess.get_inputs()[0].name

    vec_0 = sess.get_outputs()[0].name
    vec_1 = sess.get_outputs()[1].name
    vec_2 = sess.get_outputs()[2].name
    vec_3 = sess.get_outputs()[3].name
    vec_4 = sess.get_outputs()[4].name
    print(vec_0, vec_1, vec_2, vec_3, vec_4)
    topk_scores = sess.get_outputs()[5].name
    topk_inds = sess.get_outputs()[6].name
    wh_pred = sess.get_outputs()[7].name
    offset_pred = sess.get_outputs()[8].name
    rotation_pred = sess.get_outputs()[9].name

    counter = 0
    while counter < 1000:
        counter+=1
        torch.cuda.synchronize()
        start = time.time()

        input_pm_matrix = sess.get_inputs()[1].name

        if seq_num ==1:
            output = sess.run([vec_0, vec_1, vec_2, vec_3, vec_4,
                               topk_scores, topk_inds, wh_pred, offset_pred, rotation_pred],
                              {input_tensor_name: onnxruntime_input_tensor,
                               input_pm_matrix: onnxruntime_pm_tensor,
                               })

        else:
            input_ego_motion_matrix = sess.get_inputs()[2].name

            output = sess.run([vec_0, vec_1, vec_2, vec_3, vec_4,
                               topk_scores,topk_inds,wh_pred,offset_pred,rotation_pred],
                              {input_tensor_name:onnxruntime_input_tensor,
                               input_pm_matrix:onnxruntime_pm_tensor,
                               input_ego_motion_matrix:dummy_ego_motion_matrix_tensor
                               })

        torch.cuda.synchronize()
        end = time.time()
        print('latency (ms): {:.2f}'.format( (end - start) * 1000 ))


