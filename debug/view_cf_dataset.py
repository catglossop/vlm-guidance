import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import yaml
import shutil
import pickle as pkl
from tqdm import tqdm

paths = glob.glob("/home/noam/LLLwL/lcbc/data/data_annotation/cf_dataset_v2/*/*/traj_data_filtered.pkl", recursive=True)


paths = np.random.choice(paths, 10)


for path in paths:
    data = pkl.load(open(path, "rb"))
    positions = data["position"]
    positions = positions - positions[0]
    plt.plot(positions[:,0], positions[:,1])
plt.savefig("test.png")
plt.close()

# with open("../data/data_config.yaml", "r") as f:
#         data_config = yaml.safe_load(f)

# for path in tqdm(paths):
#     new_data = {}
#     dataset_name = path.split("/")[-3]
#     waypoint_spacing = data_config[dataset_name]["metric_waypoint_spacing"]
#     data = np.load(path, allow_pickle=True)
#     if "position_old" in data.keys():
#          continue
#     positions = data['position']
#     images = glob.glob(os.path.join(os.path.dirname(path),"*.jpg"))
#     idx = len(images)
#     positions_orig = positions
#     positions = positions - positions[0]
#     try:
#         cf_chunk = (positions[idx:, :] - positions[idx, :])*waypoint_spacing + positions[idx-1, :]
#         positions = np.concatenate([positions[:idx-1, :], cf_chunk])
#         positions = np.array(positions[np.sort(np.unique(positions, axis=0, return_index=True)[1])])

#         plt.plot(positions[:,0], positions[:,1])

#         positions = positions + positions_orig[0,:]
#         new_data = {key: data[key] for key in data.keys()}
#         new_data["position_old"] = positions_orig
#         new_data["position"] = positions
#         shutil.copy(path, path.replace("traj_data_filtered.pkl", "traj_data_filtered_old.pkl"))
#         with open(path, "wb") as f:
#             pkl.dump(new_data, f)
#     except:
#         print("Failed")
#         print(path)
#         breakpoint()
    