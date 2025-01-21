import os 
import glob
import shutil
import numpy as np
from tqdm import tqdm

DIR = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup"
TRAIN = 0.8
paths = glob.glob(DIR + "/*/*")

(train_paths, val_paths) = np.array_split(paths, [int(len(paths)*TRAIN)])

# Make dirs
os.makedirs(DIR + "/train", exist_ok=True)
os.makedirs(DIR + "/val", exist_ok=True)

for path in tqdm(train_paths):
    if os.path.isdir(DIR + "/train/" + path.split("/")[-1]):
        continue
    shutil.copytree(path, DIR + "/train/" + path.split("/")[-1])

for path in tqdm(val_paths):
    if os.path.isdir(DIR + "/val/" + path.split("/")[-1]):
        continue
    shutil.copytree(path, DIR + "/val/" + path.split("/")[-1])

