import os
import numpy as np
import yaml
import torch
from pyquaternion import Quaternion


def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def get_ego_pose_dict(engo_info_dict):
    ego_translation = [engo_info_dict["egoX"], engo_info_dict["egoY"], engo_info_dict["egoZ"]]
    ego_rotation = [engo_info_dict["ego_quat0"], engo_info_dict["ego_quat1"],
                    engo_info_dict["ego_quat2"], engo_info_dict["ego_quat3"], ]
    ego_dict_ = {}
    ego_dict_.update({"rotation": ego_rotation})
    ego_dict_.update({"translation": ego_translation})
    return ego_dict_


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def mat2pose_vec(matrix: torch.Tensor):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


if __name__ == '__main__':
    data_dir = "/data/zdx/Data/data_nuscene/0_403"
    item_list = os.listdir(data_dir)
    item_list.sort(key=lambda x: float(x))

    len_list = len(item_list)

    save_numpy = np.zeros([len_list, 6])

    for idx in range(len_list - 1):
        print("processing file %i" %idx)
        tmp_yaml_curr = os.path.join(data_dir,item_list[idx],"ego","ego.yaml")
        tmp_yaml_next = os.path.join(data_dir,item_list[idx + 1],"ego","ego.yaml")
        ego_info_curr = yaml.safe_load(open(tmp_yaml_curr,"r"))
        ego_info_next = yaml.safe_load(open(tmp_yaml_next,"r"))
        ego_dict_curr = get_ego_pose_dict(ego_info_curr)
        ego_dict_next = get_ego_pose_dict(ego_info_next)
        egopose_t0 = convert_egopose_to_matrix_numpy(ego_dict_curr)
        egopose_t1 = convert_egopose_to_matrix_numpy(ego_dict_next)
        future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0
        future_egomotion = torch.Tensor(future_egomotion).float()
        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        save_numpy[idx,:] = future_egomotion.numpy()

    np.save("../../data/ego_motion_nuscene.npy", save_numpy)
