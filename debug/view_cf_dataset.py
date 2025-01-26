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

paths = glob.glob("/home/noam/LLLwL/lcbc/data/data_annotation/cf_dataset_v3/*/*/traj_data_filtered.pkl", recursive=True)
# paths = random.sample(paths, 10)
# breakpoint()
# path = random.choice(paths)
# # while True:
# fig, ax = plt.subplots(1, 2)

# # for path in paths:
# data = pkl.load(open(path, "rb"))
# positions = data["position"]
# positions = positions - positions[0]
# language = data["language_annotations"][0]["traj_description"]

# image_paths = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.jpg")), key=lambda x: int(x.split("/")[-1].split(".")[0]))
# image_path = image_paths[-1]
# idx = int(image_path.strip(".jpg").split("/")[-1])
# image = np.array(Image.open(image_path))
# ax[0].plot(positions[:,0], positions[:,1])
# ax[0].plot(positions[idx:,0], positions[idx:,1], "g")
# ax[0].plot(positions[idx,0], positions[idx,1], "ro")
# ax[1].imshow(image)
# ax[1].set_title(language)
# # plt.show()
# plt.savefig("temp.png")

# # with open("../data/data_config.yaml", "r") as f:
# #         data_config = yaml.safe_load(f)
# ACTION_HORIZON = 8
for path in tqdm(paths):
    print(path)
    new_data = {}
#     dataset_name = path.split("/")[-3]
#     waypoint_spacing = data_config[dataset_name]["metric_waypoint_spacing"]
    data = np.load(path, allow_pickle=True)
    if "position_old" in data.keys():
         continue
    positions = data['position']
    images = glob.glob(os.path.join(os.path.dirname(path),"*.jpg"))
    idx = len(images)
    positions_orig = positions
    positions = positions - positions[0]
#     assert len(images) == positions.shape[0] - ACTION_HORIZON, f"len(images) = {len(images)}, positions.shape[0] = {positions.shape[0]}"
    try:
#         if positions.shape[0] == idx:
#             idx = positions.shape[0] - ACTION_HORIZON
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
        shutil.copy(path, path.replace("traj_data_filtered.pkl", "traj_data_filtered_old.pkl"))
        with open(path, "wb") as f:
          pkl.dump(new_data, f)
    except Exception as e:
        print(e)
        print("Failed")
        print(path)
        breakpoint()
    