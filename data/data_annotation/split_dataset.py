import os
import glob
import sys
import random

TRAIN = 0.8
VAL = 0.2

dataset = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/"
os.makedirs(dataset + "train", exist_ok=True)
os.makedirs(dataset + "val", exist_ok=True)

train_paths = glob.glob(dataset + "*/train/*")
val_paths = glob.glob(dataset + "*/val/*")


# for path in train_paths:
#     try:
#         os.rename(path, dataset + "train/" + path.split("/")[-1])
#     except Exception as e:
#         print(e)
#         print(path)
#         continue
for path in val_paths:
    try:
        os.rename(path, dataset + "val/" + path.split("/")[-1])
    except Exception as e:
        print(e)
        print(path)
        continue





# datasets = glob.glob("/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/*")

# for dataset in datasets:

#     dataset_paths = glob.glob(dataset + "/*")
#     breakpoint()
#     total_len = len(dataset_paths)
#     train_len = int(total_len * TRAIN)
#     val_len = int(total_len * VAL)
    
#     # shuffle paths
#     random.shuffle(dataset_paths)
#     train_paths = dataset_paths[:train_len]
#     val_paths = dataset_paths[train_len:]

#     # create train and val directories
#     train_dir = dataset + "/train"
#     val_dir = dataset + "/val"
#     os.mkdir(train_dir)
#     os.mkdir(val_dir)

#     # move files to train and val directories
#     for path in train_paths:
#         os.rename(path, train_dir + "/" + path.split("/")[-1])
#     for path in val_paths:
#         os.rename(path, val_dir + "/" + path.split("/")[-1])
    
