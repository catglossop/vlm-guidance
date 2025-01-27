import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import yaml
import shutil
import pickle as pkl
from tqdm import tqdm
from PIL import Image
import random 

paths = glob.glob("/hdd/lcbc_datasets/go_stanford2_cf/*/traj_data_filtered.pkl", recursive=True)

for path in tqdm(paths):
    print(path)
    new_data = {}
    dataset_name = path.split("/")[-3]
    traj_name = path.split("/")[-2]

    path_old = path.replace("traj_data_filtered.pkl", "traj_data_filtered_old.pkl")
    if not os.path.exists(path_old):
        shutil.copy(path, path.replace("traj_data_filtered.pkl", "traj_data_filtered_old.pkl"))
        data = np.load(path, allow_pickle=True)
    else:
        os.remove(path)
        data = np.load(path_old, allow_pickle=True)
    if "position_old" in data.keys():
         continue
    positions = data['position']
    images = glob.glob(os.path.join(os.path.dirname(path),"*.jpg"))
    idx = len(images)
    positions_orig = positions
    positions = positions - positions[0]
    try:
        # print(traj_name)
        # plt.close()
        # plt.plot(positions[:,0], positions[:,1])
        # plt.plot(positions[idx:,0], positions[idx:,1], "g")
        # plt.savefig(f"orig_positions_{traj_name}.png")
        # breakpoint()
        cf_chunk = (positions[idx:, :] - positions[idx, :]) + positions[idx-1, :]
        positions = np.concatenate([positions[:idx-1, :], cf_chunk])
        positions = np.array(positions[np.sort(np.unique(positions, axis=0, return_index=True)[1])])
        # plt.close()
        # plt.plot(positions[:,0], positions[:,1])
        # plt.plot(positions[idx:,0], positions[idx:,1], "g")
        # plt.savefig("changed_positions.png")
        # breakpoint()
        positions = positions + positions_orig[0,:]
        new_data = {key: data[key] for key in data.keys()}
        new_data["position_old"] = positions_orig
        new_data["position"] = positions
        with open(path, "wb") as f:
          pkl.dump(new_data, f)
    except Exception as e:
        print(e)
        print("Failed")
        print(path)
        breakpoint()
    