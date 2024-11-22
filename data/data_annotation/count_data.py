import os
import glob
import numpy as np


paths = glob.glob("/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup/*/*", recursive=True)

print("Number of trajectories: ", len(paths))

imgs = glob.glob("/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup/*/*/*.jpg", recursive=True)

print("Number of frames: ", len(imgs))
total_num_annotations = 0
for path in paths:
    if not os.path.exists(path + "/traj_data.pkl"):
        continue
    traj_data = np.load(path + "/traj_data.pkl", allow_pickle=True)
    total_num_annotations += len(traj_data["language_annotations"])

print("Number of annotations: ", total_num_annotations)
