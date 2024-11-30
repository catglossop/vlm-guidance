import glob
import numpy
import os

subdatasets = glob.glob("/home/noam/LLLwL/datasets/atomic_dataset/*")

# remove old subdirs
for subdataset in subdatasets:
    if os.path.exists(f"{subdataset}/turn_right"):
        os.system(f"rm -r {subdataset}/turn_right")
    if os.path.exists(f"{subdataset}/turn_left"):
        os.system(f"rm -r {subdataset}/turn_left")
    if os.path.exists(f"{subdataset}/go_forward"):
        os.system(f"rm -r {subdataset}/go_forward")
    if os.path.exists(f"{subdataset}/stop"):
        os.system(f"rm -r {subdataset}/stop")
for subdataset in subdatasets:
    print(subdataset)
    turn_right = glob.glob(f"{subdataset}/*/turn_right.txt")
    turn_left = glob.glob(f"{subdataset}/*/turn_left.txt")
    go_forward = glob.glob(f"{subdataset}/*/go_forward.txt")
    stop = glob.glob(f"{subdataset}/*/stop.txt")

    print("num turn right: ", len(turn_right))
    print("num turn left: ", len(turn_left))
    print("num go forward: ", len(go_forward))
    print("num stop: ", len(stop))
    print("total: ", len(turn_right) + len(turn_left) + len(go_forward) + len(stop))

    # make subdirs
    os.makedirs(f"{subdataset}/turn_right", exist_ok=True)
    os.makedirs(f"{subdataset}/turn_left", exist_ok=True)
    os.makedirs(f"{subdataset}/go_forward", exist_ok=True)
    os.makedirs(f"{subdataset}/stop", exist_ok=True)
    # move files
    for file in turn_right:
        folder = file.split("/")[-2]
        curr_folder = ("/").join(file.split("/")[:-1])
        new_folder = f"{subdataset}/turn_right/{folder}"
        os.rename(curr_folder, new_folder)
    for file in turn_left:
        folder = file.split("/")[-2]
        curr_folder = ("/").join(file.split("/")[:-1])
        new_folder = f"{subdataset}/turn_left/{folder}"
        os.rename(curr_folder, new_folder)
    for file in go_forward:
        folder = file.split("/")[-2]
        curr_folder = ("/").join(file.split("/")[:-1])
        new_folder = f"{subdataset}/go_forward/{folder}"
        os.rename(curr_folder, new_folder)
    for file in stop:
        folder = file.split("/")[-2]
        curr_folder = ("/").join(file.split("/")[:-1])
        new_folder = f"{subdataset}/stop/{folder}"
        os.rename(curr_folder, new_folder)
    
    breakpoint()
    