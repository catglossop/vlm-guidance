import os 
import glob
import shutil
import numpy as np
from tqdm import tqdm

# DIR = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup"
DIR = "/hdd/cf_v2_dataset"
dataset_name = "cf_v2_dataset"
TRAIN = 0.8
RERUN = True
paths = glob.glob(DIR + "/*/*")

paths = [path for path in paths if "train" not in path and "val" not in path]

(train_paths, val_paths) = np.array_split(paths, [int(len(paths)*TRAIN)])

if RERUN:
    if os.path.exists(DIR + "/train"):
        shutil.rmtree(DIR + "/train")
    if os.path.exists(DIR + "/val"):
        shutil.rmtree(DIR + "/val")

# Make dirs
os.makedirs(DIR + f"/train/{dataset_name}", exist_ok=True)
os.makedirs(DIR + f"/val/{dataset_name}", exist_ok=True)

for path in tqdm(train_paths):
    if os.path.isdir(DIR + f"/train/{dataset_name}/" + path.split("/")[-1]):
        continue
    try:
        shutil.copytree(str(path), DIR + f"/train/{dataset_name}/" + str(path.split("/")[-1]))
    except Exception as e:
        print(e)
        breakpoint()

for path in tqdm(val_paths):
    if os.path.isdir(DIR + f"/val/{dataset_name}/" + path.split("/")[-1]):
        continue
    shutil.copytree(str(path), DIR + f"/val/{dataset_name}/" + path.split("/")[-1])

