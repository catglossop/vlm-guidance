import os
import glob
import shutil 
import tqdm
import numpy as np

dataset_path = "/hdd/outdoor_labelled/go_stanford2_labelled/"
output_path = "/hdd/go_stanford2_labelled_cleaned/"

paths = glob.glob(dataset_path + "*")
os.makedirs(output_path, exist_ok=True)
print(len(paths))
removed_cnt = 0
while len(paths) > 0:
    path = paths.pop()
    path_parts = path.split("/")[-1].split("_chunk_")
    path_root = path_parts[0]

    # check if there are multiple chunks of same chunk number 
    path_chunk_num = path_parts[-1].split("_")[0]
    paths_same_chunks = glob.glob(dataset_path + path_root + "_chunk_" + str(path_chunk_num) + "_*")
    try:
        keep_path = paths_same_chunks[0]
    except:
        breakpoint()
    if len(paths_same_chunks) > 1:
        for p in paths_same_chunks:
            try:
                paths.remove(p)
            except Exception as e:
                print(e)
                pass
        if os.path.isdir(output_path + keep_path.split("/")[-1]):
            continue
        shutil.copytree(keep_path, output_path + keep_path.split("/")[-1])
    else:
        if os.path.isdir(output_path + keep_path.split("/")[-1]):
            continue
        shutil.copytree(keep_path, output_path + keep_path.split("/")[-1])
    removed_cnt += len(paths_same_chunks) - 1
    
    # Check if there are multiple chunks with same start and end number
    start_end = ("_").join(path_parts[-1].split("_")[-4:])
    paths_same_idxs = glob.glob(dataset_path + path_root + "_chunk_*" + start_end)
    try:
        keep_path = paths_same_idxs[0]
    except:
        breakpoint()
    if len(paths_same_idxs) > 1:
        for p in paths_same_idxs:
            try:
                paths.remove(p)
            except Exception as e:
                print(e)
                pass

        if os.path.isdir(output_path + keep_path.split("/")[-1]):
            continue
        shutil.copytree(keep_path, output_path + keep_path.split("/")[-1])
    else:
        if os.path.isdir(output_path + keep_path.split("/")[-1]):
            continue
        shutil.copytree(keep_path, output_path + keep_path.split("/")[-1])
    removed_cnt += len(paths_same_idxs) - 1

    print("Removed: ", removed_cnt)
    print("Remaining: ", len(paths))