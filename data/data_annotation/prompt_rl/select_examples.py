import glob
from PIL import Image
import os 
import numpy as np
import random


# Get all files from gnm_dataset
traj_paths = glob.glob("/home/noam/LLLwL/datasets/gnm_dataset/*/*", recursive=True)
traj_paths = [traj for traj in traj_paths if os.path.isdir(traj)]
stop = False
while not stop:
    random_traj = random.choice(traj_paths)
    print("Current trajectory: ", random_traj)
    image_paths = sorted(glob.glob(random_traj + "/*.jpg"), key=lambda x: int(x.split("/")[-1].split(".")[0]))
    start_idx = 0
    end_idx = np.min([len(image_paths), start_idx + random.randint(30, 40)])
    while end_idx - start_idx > 5:
        images = [Image.open(img) for img in image_paths[start_idx:end_idx]]
        images[0].save("out.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
        os.system("open out.gif")
        use = input("Use this trajectory? (y/n): ")
        if use == "y":
            labels = []
            name = input("Name of trajectory: ")
            os.system(f"mkdir examples/{name}")
            for i, img in enumerate(images):
                img.save(f"examples/{name}/{i}.png")
            label = input("Label of trajectory: ")
            labels.append(label)
            next_label = input("Add another label? (y/n): ")
            while next_label == "y":
                label = input("Label of trajectory: ")
                labels.append(label)
                next_label = input("Add another label? (y/n): ")
            with open(f"examples/{name}/labels.txt", "w") as f:
                f.write("\n".join(labels))
        stop = input("Stop? (y/n): ") == "y"
        start_idx = end_idx
        end_idx = np.min([len(image_paths), start_idx + random.randint(20, 25)])

