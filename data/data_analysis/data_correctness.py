import glob 
import os
import numpy as np
from PIL import Image as PILImage
import pickle as pkl
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2

DATA_PATH = ["/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/cory_hall_labelled/", 
             "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/go_stanford_cropped_labelled/",
             "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/sacson_labelled/"]
TRAJ_CRITERIA_CHECK = "A trajectory is correct if it has:\n 1) Correct objects/structures,\n 2) Correct motion,\n 3) Correct grounding (locations of objects/structures),\n 4) Is descriptive (if there is something that is useful for navigation, it should be present in the instruction).\n Is the trajectory correct? (y/n) \n"
TRAJ_CRITERIA_INCORRECT = "Please provide the reason why the trajectory is incorrect:\n 1 - Incorrect objects/structures,\n 2 - Incorrect motion,\n 3 - Incorrect grounding,\n 4 - Not descriptive,\n seperated by a comma (e.g. 1,2,3)"
subresult = {"correct": 0, "incorrect": 0, "incorrect_reasons": {1: 0, 2: 0, 3: 0, 4: 0}}
reason_map = {1: "Incorrect objects/structures", 2: "Incorrect motion", 3: "Incorrect grounding", 4: "Not descriptive"}

def create_gif_from_path(path):
    images = [PILImage.open(f) for f in sorted(glob.glob(path + "/*.jpg"), key=lambda x: int(x.split('/')[-1].split('.')[0]))]
    images[0].save("trajectory.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
    videodims = (100,100)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')    
    video = cv2.VideoWriter("trajectory.mp4",fourcc, 60)
    #draw stuff that goes on every frame here
    for img in images:
        imtemp = img.copy()
        # draw frame specific stuff here.
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()

def load_data_from_path(path):
    traj_data = pkl.load(open(path + "/traj_data.pkl", "rb"))
    return traj_data

# Load dataset 
def load_dataset():
    paths = []
    dataset_lengths = {}
    for dataset_path in DATA_PATH:
        dataset_name = dataset_path.split("/")[-2]
        subdataset_paths = glob.glob(dataset_path + "/*")
        if os.path.exists(dataset_path + "current_state"):
            subdataset_paths.remove(dataset_path + "current_state")
        print("Found {} subdatasets in {}".format(len(subdataset_paths), dataset_path))
        dataset_lengths[dataset_name] = len(subdataset_paths)
        paths.extend(subdataset_paths)
    print("Found {} trajectories".format(len(paths)))
    return paths, dataset_lengths

def check_results():
    checked_paths = []
    for dataset in DATA_PATH:
        subchecked_paths = glob.glob(f"{dataset}/*/analysis.pkl", recursive=True)
        checked_paths.extend(subchecked_paths)
    
    num_checked = len(checked_paths)

    results = {}
    for dataset in DATA_PATH:
        dataset_name = dataset.split("/")[-2]
        results[dataset_name] = deepcopy(subresult)

    for path in tqdm(checked_paths):
        dataset_name = path.split("/")[-3]
        with open(path, "rb") as f:
            traj_results = pkl.load(f)
        traj_results = traj_results["results"]
        for traj in traj_results:
            if traj["correct"]:
                results[dataset_name]["correct"] += 1
            else:
                results[dataset_name]["incorrect"] += 1
                for reason in traj["incorrect_reasons"]:
                    results[dataset_name]["incorrect_reasons"][reason] += traj["incorrect_reasons"][reason]
    
    # Sum up results over all datasets
    total_results = {"correct": [], "incorrect": [], "incorrect_reasons": {1: [], 2: [], 3: [], 4: []}}
    for dataset in results.keys():
        total_results["correct"].append(results[dataset]["correct"])
        total_results["incorrect"].append(results[dataset]["incorrect"])
        for reason in results[dataset]["incorrect_reasons"]:
            total_results["incorrect_reasons"][reason].append(results[dataset]["incorrect_reasons"][reason])
    total_results["correct"].insert(0, np.sum(total_results["correct"]))
    total_results["incorrect"].insert(0, np.sum(total_results["incorrect"]))
    for reason in total_results["incorrect_reasons"]:
        total_results["incorrect_reasons"][reason].insert(0, np.sum(total_results["incorrect_reasons"][reason]))
    incorrect_reasons_temp = total_results["incorrect_reasons"]
    incorrect_reasons = {}
    for reason in incorrect_reasons_temp:
        incorrect_reasons[reason_map[reason]] = incorrect_reasons_temp[reason]
    del total_results["incorrect_reasons"]
    total_results.update(incorrect_reasons)

    # Visualize results
    dataset_names = ["Total"] + list(results.keys()) 
    print("Dataset names: ", dataset_names)
    x = np.arange(len(dataset_names))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(20, 10))

    for metric, value in total_results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=metric)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Num. of instr.')
    ax.set_title('Data Analysis Results')
    ax.set_xticks(x + width, dataset_names)
    ax.legend(loc='upper left', ncols=3)

    # Print results
    print("Results:")
    print("=====================================")
    print("Total:")
    print(f"Correct: {total_results['correct']} ({total_results['correct'][0] / (total_results['correct'][0] + total_results['incorrect'][0]) * 100}%)")
    print(f"Incorrect: {total_results['incorrect']} ({total_results['incorrect'][0] / (total_results['correct'][0] + total_results['incorrect'][0]) * 100}%)")
    for reason in total_results:
        if reason == "correct" or reason == "incorrect":
            continue
        print(f"Reason {reason}: {total_results[reason]} ({total_results[reason][0] / total_results['incorrect'][0] * 100}%)")
    print("-------------------------------------")
    for dataset in results.keys():
        print(f"Dataset: {dataset}")
        print(f"Correct: {results[dataset]['correct']}")
        print(f"Incorrect: {results[dataset]['incorrect']}")
        for reason in results[dataset]["incorrect_reasons"]:
            print(f"Reason {reason}: {results[dataset]['incorrect_reasons'][reason]}")
        print("-------------------------------------")

    plt.show()

# Function for looping through paths
def main(args):
    paths, _ = load_dataset()
    if args.reevaluate:
        for path in paths:
            if os.path.exists(path + "/analysis.pkl"):
                os.remove(path + "/analysis.pkl")
            if os.path.exists(path + "/traj_stats.pkl"):
                os.remove(path + "/traj_stats.pkl")
    contd = True
    num_trajs_checked = 0
    for path in paths:
        traj_analysis = {"results": []}
        print("Path: ", path)
        if not contd or num_trajs_checked >= args.num_samples:
            print("[Stopping data analysis]")
            break
        if os.path.exists(path + "/analysis.pkl") and not args.reevaluate:
            print("Path already analyzed, skipping")
            num_trajs_checked += 1
            print("Number of trajectories checked: ", num_trajs_checked)
            continue

        dataset_name = path.split("/")[-2]
        traj_data = load_data_from_path(path)
        create_gif_from_path(path)
        print("Number of annotations: ", len(traj_data["language_annotations"]))
        print("Trajectory descriptions: ")
        for i, desc in enumerate(traj_data["language_annotations"]):
            instr_analysis = deepcopy(subresult)
            print(i, desc)
            
            check = input(TRAJ_CRITERIA_CHECK)
            if check == "y":
                instr_analysis["correct"] = 1
                traj_analysis["results"].append(instr_analysis) 
            elif check == "n":
                instr_reasons = input(TRAJ_CRITERIA_INCORRECT)
                instr_reasons = instr_reasons.split(",")
                instr_reasons = [int(reason) for reason in instr_reasons]
                instr_analysis["incorrect"] = 1
                for reason in instr_reasons:
                    instr_analysis["incorrect_reasons"][reason] = 1
                traj_analysis["results"].append(instr_analysis)
            else:
                print("Invalid input")
                continue
        with open(path + "/analysis.pkl", "wb") as f:
            pkl.dump(traj_analysis, f)
        num_trajs_checked += 1
        print("Number of trajectories checked: ", num_trajs_checked)
        contd = input("Continue? (y/n)")
        if contd == "n":
            contd = False

    print("Finished")
    if args.verbose:
        check_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()
    main(args)


