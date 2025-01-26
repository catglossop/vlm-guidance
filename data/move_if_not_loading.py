import glob
import os
import shutil 
import numpy as np

paths = glob.glob("/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup/train/*/*/traj_data_filtered.pkl")
breakpoint()
cnt_broken = 0
for path in paths:
    print(cnt_broken)
    try:
        data = np.load(path, allow_pickle=True)
    except:
        cnt_broken += 1
        print(path)
        # breakpoint()
        os.remove(path)
        shutil.copyfile(path.replace("traj_data_filtered.pkl", "traj_data_w_embed_t5.pkl"), path)
