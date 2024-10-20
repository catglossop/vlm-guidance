import glob 
import os
import numpy as np
from PIL import Image as PILImage
import pickle as pkl

DATA_PATH = ["/home/noam/LLLwL/lcbc/data/data_annotation/cory_hall_labelled/", 
             "/home/noam/LLLwL/lcbc/data/data_annotation/go_stanford_cropped_labelled/",
             "/home/noam/LLLwL/lcbc/data/data_annotation/sacson_labelled/"]
REEVALUATE = False
def create_gif_from_path(path):
    images = [PILImage.open(f) for f in sorted(glob.glob(path + "/*.jpg"), key=lambda x: int(x.split('/')[-1].split('.')[0]))]
    images[0].save("trajectory.gif", save_all=True, append_images=images[1:], duration=100, loop=0)

def load_data_from_path(path):
    traj_data = pkl.load(open(path + "/traj_data.pkl", "rb"))
    return traj_data

# Load dataset 
paths = []
dataset_lengths = {}
for dataset_path in DATA_PATH:
    dataset_name = dataset_path.split("/")[-2]
    subdataset_paths = glob.glob(dataset_path + "/*")
    subdataset_paths.remove(dataset_path + "current_state.pkl")
    print("Found {} subdatasets in {}".format(len(subdataset_paths), dataset_path))
    dataset_lengths[dataset_name] = len(subdataset_paths)
    paths.extend(subdataset_paths)
print("Found {} trajectories".format(len(paths)))

# Function for looping through paths
dataset_dict = {}
for dataset in DATA_PATH:
    dataset_name = dataset.split("/")[-2]
    dataset_dict[dataset_name] = {"num_correct": 0, "num_total": 0}
contd = True
num_trajs_checked = 0
for path in paths:
    print("Path: ", path)
    if not contd:
        print("[Stopping data analysis]")
        break
    if os.path.exists(path + "/traj_stats.pkl") and not REEVALUATE:
        traj_stats = pkl.load(open(path + "/traj_stats.pkl", "rb"))
        print("Path already analyzed, skipping")
        dataset_name = path.split("/")[-2]
        dataset_dict[dataset_name]["num_correct"] += traj_stats["num_correct"]
        dataset_dict[dataset_name]["num_total"] += traj_stats["num_total"]
        num_trajs_checked += 1
        print("Number of trajectories checked: ", num_trajs_checked)
        with open("overall_stats.pkl", "wb") as f:
            pkl.dump(dataset_dict, f)
        continue

    traj_stats = {"num_correct": 0, "num_total": 0}
    dataset_name = path.split("/")[-2]
    traj_data = load_data_from_path(path)
    create_gif_from_path(path)
    print("Trajectory descriptions: ")
    for i, desc in enumerate(traj_data["language_annotations"]):
        print(i, desc)
    print("Number of annotations: ", len(traj_data["language_annotations"]))
    
    num_correct = input("How many annotations correctly describe the trajectory?")
    try: 
        num_correct = int(num_correct)
        if num_correct < 0 or num_correct > len(traj_data["language_annotations"]):
            print("Invalid input")
            continue
    except ValueError:
        print("Invalid input")
        continue
    traj_stats["num_correct"] = num_correct
    traj_stats["num_total"] = len(traj_data["language_annotations"])
    with open(path + "/traj_stats.pkl", "wb") as f:
        pkl.dump(traj_stats, f)
    dataset_dict[dataset_name]["num_correct"] += num_correct
    dataset_dict[dataset_name]["num_total"] += len(traj_data["language_annotations"])
    num_trajs_checked += 1
    print("Number of trajectories checked: ", num_trajs_checked)
    contd = input("Continue? (y/n)")
    if contd == "n":
        contd = False
    with open("overall_stats.pkl", "wb") as f:
        pkl.dump(dataset_dict, f)

print("Finished")

# Print out results
total_correct = 0
total_length = 0
for dataset_name in dataset_dict.keys():
    print("--------------------")
    print("Dataset: ", dataset_name)

    if dataset_dict[dataset_name]["num_total"] == 0:
        print(f"No annotations for dataset: {dataset_name}")
        continue
    num_correct = dataset_dict[dataset_name]["num_correct"]
    print("Number of correct annotations: ", num_correct)
    print("Number of annotations: ", dataset_lengths[dataset_name])
    print("Percent correct: ", num_correct / dataset_dict[dataset_name]["num_total"])
    print("--------------------")
    total_correct += num_correct
    total_length += dataset_dict[dataset_name]["num_total"]
print("================================")
print("Total number of correct annotations: ", total_correct)
print("Total number of annotations: ", total_length)
print("Total percent correct: ", total_correct / total_length)


