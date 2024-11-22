import pickle as pkl
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
import glob
from tqdm import tqdm
from PIL import Image


MAX_STEPS = 20
DATASET_PATHS = "/home/noam/LLLwL/datasets/gnm_dataset"
VISUALIZE = False
DEBUG = False
TURN_THRESHOLD = 1.0 # 45 degrees
STOP_THRESHOLD = 0.5
OUTPUT_PATH = "/home/noam/LLLwL/datasets/gnm_dataset_atomic_lang"
base_instructions = ["Turn left", "Turn right", "Go forward", "Stop"]
varied_forward = [
    "Move forward",
    "Proceed straight",
    "Continue ahead",
    "Keep going forward",
    "Advance",
    "Go straight on",
    "Head straight",
    "March forward",
    "Push forward",
    "Progress forward",
    "Forge ahead",
    "Maintain your course",
    "Stay on track",
    "Keep moving ahead",
    "Drive straight ahead",
    "Press on",
    "Plow ahead",
    "Move ahead",
    "Continue on your path",
    "Sustain your direction"
]
varied_right = [
    "Make a right turn",
    "Veer to the right",
    "Head rightward",
    "Take a right here",
    "Go right at the next opportunity",
    "Bear right",
    "Swing to the right",
    "Proceed to the right",
    "Angle right",
    "Shift right",
    "Rotate right",
    "Pivot to the right",
    "Steer right",
    "Divert to the right",
    "Bank right",
    "Curve right",
    "Move to the right side",
    "Navigate right",
    "Aim right",
    "Adjust your path to the right"
]
varied_left = [
    "Make a left turn",
    "Veer to the left",
    "Head leftward",
    "Take a left here",
    "Go left at the next opportunity",
    "Bear left",
    "Swing to the left",
    "Proceed to the left",
    "Angle left",
    "Shift left",
    "Rotate left",
    "Pivot to the left",
    "Steer left",
    "Divert to the left",
    "Bank left",
    "Curve left",
    "Move to the left side",
    "Navigate left",
    "Aim left",
    "Adjust your path to the left"
]

varied_stop = [
    "cease",
    "halt",
    "desist",
    "terminate",
    "end",
    "quit",
    "suspend",
    "discontinue",
    "abandon",
    "forbear",
    "pause",
    "ceasefire",
    "standstill",
    "break off",
    "conclude",
    "finish",
    "terminate",
    "cease and desist",
    "bring to a halt",
    "put an end to"
]

def get_yaw_delta(yaw_1, yaw_2):
    yaw_delta = yaw_2 - yaw_1
    # print(f"Yaw delta: {yaw_delta}")
    # breakpoint()
    # yaw_delta_sign = -1 if yaw_delta >= np.pi else 1
    # yaw_delta = yaw_delta + yaw_delta_sign*2*np.pi
    return yaw_delta

def get_language_instructions(path):
    global TURN_THRESHOLD, STOP_THRESHOLD, MAX_STEPS, DEBUG
    chunk_idx = 0
    i = 0
    total_traj_len = len(glob.glob(path + "/*.jpg"))
    if not os.path.isdir(path):
        return
    with open(path + "/traj_data.pkl", "rb") as f:
        traj_data = pkl.load(f) 
    
    yaw = traj_data["yaw"]
    pos = traj_data["position"]

    while i < total_traj_len:
        # Get traj range
        curr_traj_len = 1
        while i+curr_traj_len < total_traj_len and np.abs(get_yaw_delta(yaw[i], yaw[i+curr_traj_len])) < TURN_THRESHOLD and curr_traj_len < MAX_STEPS:
            curr_traj_len += 1
        if i+curr_traj_len >= total_traj_len:
            break
        if get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) > TURN_THRESHOLD:
            if DEBUG:
                print("Result is turn left")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
            language_instruction = base_instructions[0]
            varied_language_instruction = random.choice(varied_left)
        elif get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) < -TURN_THRESHOLD:
            if DEBUG:
                print("Result is turn right")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
            language_instruction = base_instructions[1]
            varied_language_instruction = random.choice(varied_right)
        elif np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1)) > STOP_THRESHOLD:
            if DEBUG:
                print("Result is go forward")
                print(f"Distance: {np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1))}")
            language_instruction = base_instructions[2]
            varied_language_instruction = random.choice(varied_forward)
        else:
            if DEBUG:
                print("Result is stop")
                print(f"Distance: {np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1))}")
            language_instruction = base_instructions[3]
            varied_language_instruction = random.choice(varied_stop)
        
        curr_traj_data = {key: val[i:i+curr_traj_len] for key, val in traj_data.items()}
        curr_traj_data["language_instruction"] = language_instruction
        curr_traj_data["varied_language_instruction"] = varied_language_instruction
        curr_output_path = os.path.join(OUTPUT_PATH, f"{('/').join(path.split('/')[-2:])}_chunk_{chunk_idx}")
        os.makedirs(curr_output_path, exist_ok=True)
        with open( curr_output_path + f"/traj_data.pkl", "wb") as f:
            pkl.dump(curr_traj_data, f)
        with open( curr_output_path + f"/{('_').join(language_instruction.lower().split(' '))}.txt", "w") as f:
            f.write(language_instruction)
        for j in range(i, i+curr_traj_len):
            image = iio.imread(os.path.join(path, f"{j}.jpg"))
            iio.imwrite(curr_output_path + f"/{j}.jpg", image)  
        if DEBUG:
            images = [Image.open(os.path.join(curr_output_path, f"{j}.jpg")) for j in range(i, i+curr_traj_len)]
            images[0].save("trajectory.gif", save_all=True, append_images=images[1:], duration=100, loop=1)
            breakpoint()
        i += curr_traj_len
        chunk_idx += 1 

def main(paths):
    for path in tqdm(paths):
        if DEBUG:
            print(f'Processing {path}')
        get_language_instructions(path)

if __name__ == "__main__":
    paths = glob.glob(f"{DATASET_PATHS}/*/*", recursive=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True) 
    main(paths)






