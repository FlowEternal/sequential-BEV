import numpy as np

# ego motion for nuscene
ego_motion_np = np.load("../../data/ego_motion_nuscene.npy", allow_pickle=True)
target_output = open("../../data/ego_motion_nuscene.txt", "w")

for idx, tmp_ in enumerate(ego_motion_np):
    target_output.writelines("frame %i %f %f %f %f %f %f\n" %(idx,tmp_[0],tmp_[1],tmp_[2],tmp_[3],tmp_[4],tmp_[5]) )
