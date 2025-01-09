import os
import glob
import sys
import random
import shutil
from tqdm import tqdm

TRAIN = 0.8
VAL = 0.2
RERUN = True
output_dir = "/home/noam/LLLwL/datasets/atomic_dataset_sorted"
dataset = "/home/noam/LLLwL/datasets/atomic_dataset_fixed/"
atomic_names = ["turn_left", "turn_right", "go_forward", "stop"]
if RERUN:
    for atomic_name in atomic_names:
        os.system(f"rm -r {output_dir}_{atomic_name}")
        os.makedirs(f"{output_dir}_{atomic_name}/train")
        os.makedirs(f"{output_dir}_{atomic_name}/val")
else:
    for atomic_name in atomic_names:
        os.makedirs(f"{output_dir}_{atomic_name}/train")
        os.makedirs(f"{output_dir}_{atomic_name}/val")

for atomic_name in atomic_names:
    print(f"Processing {atomic_name}")
    atomic_output_dir = output_dir + f"_{atomic_name}/"
    all_paths = glob.glob(dataset + f"*/{atomic_name}/*", recursive=True)
    random.shuffle(all_paths)

    train_paths = all_paths[:int(len(all_paths) * TRAIN)]
    val_paths = all_paths[int(len(all_paths) * TRAIN):]
    print(f"Train: {len(train_paths)}")
    for path in tqdm(train_paths):
        try:
            shutil.copytree(path, atomic_output_dir + "train/" + path.split("/")[-1])
        except Exception as e:
            print(e)
            print(path)
            continue
    print(f"Val: {len(val_paths)}")
    for path in tqdm(val_paths):
        try:
            shutil.copytree(path, atomic_output_dir + "val/" + path.split("/")[-1])
        except Exception as e:
            print(e)
            print


    
