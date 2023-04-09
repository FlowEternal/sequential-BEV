import json
import numpy as np

import torch

from bev_vector.devkit.data.dataset import HDMapNetDataset
from bev_vector.devkit.data.rasterize import rasterize_map
from nuscenes.utils.splits import create_splits_scenes


class HDMapNetEvalDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, eval_set, result_path, thickness, max_line_count=100, max_channel=3,
                 xbound=None, ybound=None):
        if ybound is None:
            ybound = [-15., 15., 0.15]
        if xbound is None:
            xbound = [-30., 30., 0.15]

        super(HDMapNetEvalDataset, self).__init__(version, dataroot, xbound, ybound)

        scenes = create_splits_scenes()[eval_set]
        with open(result_path, 'r') as f:
            self.prediction = json.load(f)
        self.samples = [samp for samp in self.nusc.sample if self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        self.max_line_count = max_line_count
        self.max_channel = max_channel
        self.thickness = thickness

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        gt_vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])

        gt_map, _ , _ = rasterize_map(gt_vectors, self.patch_size, self.canvas_size, self.max_channel, self.thickness)
        if self.prediction['meta']['vector']:
            pred_vectors = self.prediction['results'][rec['token']]
            pred_map, confidence_level, _ = rasterize_map(pred_vectors, self.patch_size, self.canvas_size, self.max_channel, self.thickness)
        else:
            pred_map = np.array(self.prediction['results'][rec['token']]['map'])
            confidence_level = self.prediction['results'][rec['token']]['confidence_level']

        confidence_level = torch.tensor(confidence_level + [-1] * (self.max_line_count - len(confidence_level)))

        return pred_map, confidence_level, gt_map


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate nuScenes local HD Map Construction Results.')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--eval_set', type=str, default='mini_val', choices=['train', 'val', 'test', 'mini_train', 'mini_val'])

    args = parser.parse_args()

    dataset = HDMapNetEvalDataset(args.version, args.dataroot, args.eval_set, args.result_path, thickness=2)
    for i in range(dataset.__len__()):
        pred_vectors, confidence_level, gt_vectors = dataset.__getitem__(i)
