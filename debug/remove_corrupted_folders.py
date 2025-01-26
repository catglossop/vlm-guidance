import glob
import os
import shutil 
import time
from tqdm import tqdm

base_dir = "/home/noam/LLLwL/lcbc/data/data_annotation/cf_dataset_v2"

new_folder = "/hdd/cf_v2_dataset"

os.makedirs(new_folder, exist_ok=True)

paths = glob.glob(base_dir + "/*/*")

ok_cnt = 0
for path in tqdm(paths): 
    files = glob.glob(path + "/*")
    mtimes = [time.ctime(os.path.getmtime(file)) for file in files]
    not_edited = ["Jan 22" not in mtime for mtime in mtimes]
    if all(not_edited):
        traj_folder = path.split("/")[-2:]
        new_folder_path = os.path.join(base_dir, new_folder, traj_folder[0], traj_folder[1])
        shutil.copytree(path, new_folder_path)
        # print(f"Moved {path} to {os.path.join(base_dir, new_folder_path)}")
        ok_cnt += 1

print(f"Moved {ok_cnt} folders of {len(paths)}")
