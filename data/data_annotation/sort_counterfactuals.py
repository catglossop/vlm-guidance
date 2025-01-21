import os 
import glob

root = "/home/noam/LLLwL/lcbc/data/data_annotation"
cory_hall_paths = os.listdir(root + "/lcbc_datasets_backup/cory_hall_labelled")
sacson_paths = os.listdir(root + "/lcbc_datasets_backup/sacson_labelled")
scand_paths = os.listdir(root + "/lcbc_datasets_backup/scand_labelled")
go_stanford_paths = os.listdir(root + "/lcbc_datasets_backup/go_stanford_cropped_labelled")


os.makedirs(root + "/cf_dataset/cory_hall_cf", exist_ok=True)
os.makedirs(root + "/cf_dataset/sacson_cf", exist_ok=True)
os.makedirs(root + "/cf_dataset/scand_cf", exist_ok=True)
os.makedirs(root + "/cf_dataset/go_stanford_cf", exist_ok=True)

cf_paths = glob.glob(root + "/cf_dataset/*")

breakpoint()

for path in cf_paths: 
    traj_dir = path.split("/")[-1]
    traj_dir_mod = ("_").join(traj_dir.split("_")[:-2])
    if traj_dir_mod in cory_hall_paths:
        os.rename(path, root + "/cf_dataset/cory_hall_cf/" + traj_dir)
    elif traj_dir_mod in sacson_paths:
        os.rename(path, root + "/cf_dataset/sacson_cf/" + traj_dir)
    elif traj_dir_mod in scand_paths:
        os.rename(path, root + "/cf_dataset/scand_cf/" + traj_dir)
    elif traj_dir_mod in go_stanford_paths:
        os.rename(path, root + "/cf_dataset/go_stanford_cf/" + traj_dir)
    else:
        print("No match found")

