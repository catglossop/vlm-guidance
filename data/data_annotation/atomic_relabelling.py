import pickle as pkl
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
import glob
from tqdm import tqdm
from PIL import Image


MAX_STEPS = 10
DATASET_PATHS = "/home/noam/LLLwL/datasets/gnm_dataset"
VISUALIZE = False
DEBUG = False
MIN_TURN_THRESHOLD = 1.05 # 60 degrees
MIN_FORWARD_THRESHOLD = 0.8
STOP_THRESHOLD = 0.5
USE_CENTERPOINT = True
OUTPUT_PATH = "/home/noam/LLLwL/datasets/atomic_dataset_finer"
base_instructions = ["Turn left", "Turn right", "Go forward", "Stop", "Adjust left", "Adjust right"]
varied_forward = [
    "Move forward",
    "Go straight",
    "Continue forward",
    "Keep going forward",
    "Go straight on",
    "Head straight",
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
varied_mid_right = [
    "Adjust slightly to the right.",
    "Veer gently to the right.",
    "Shift your heading slightly rightward.",
    "Angle a little to the right.",
    "Drift slightly right.",
    "Make a minor adjustment to the right.",
    "Pivot gently toward the right.",
    "Lean slightly to the right.",
    "Move subtly toward the right.",
    "Curve gently to the right.",
    "Nudge a bit to the right.",
    "Adjust your course slightly rightward.",
    "Ease a little to the right.",
    "Tilt slightly toward the right.",
    "Align a fraction to the right.",
    "Shift course marginally right.",
    "Reorient gently to the right.",
    "Steer just a touch to the right.",
    "Guide your path slightly right.",
    "Correct course lightly to the right."
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
varied_mid_left = [
    "Adjust slightly to the left.",
    "Veer gently to the left.",
    "Shift your heading slightly leftward.",
    "Angle a little to the left.",
    "Drift slightly left.",
    "Make a minor adjustment to the left.",
    "Pivot gently toward the left.",
    "Lean slightly to the left.",
    "Move subtly toward the left.",
    "Curve gently to the left.",
    "Nudge a bit to the left.",
    "Adjust your course slightly leftward.",
    "Ease a little to the left.",
    "Tilt slightly toward the left.",
    "Align a fraction to the left.",
    "Shift course marginally left.",
    "Reorient gently to the left.",
    "Steer just a touch to the left.",
    "Guide your path slightly left.",
    "Correct course lightly to the left."
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
    "Rotate left",
    "Pivot to the left",
    "Steer left",
    "Divert to the left",
    "Bank left",
    "Move to the left"
    "Navigate left",
    "Aim left",
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
    # Get the two cases of yaw delta
    yaw_delta_init = yaw_2 - yaw_1
    if (yaw_delta_init > 0):
        yaw_delta_wrap = yaw_delta_init - 2*np.pi
    else:
        yaw_delta_wrap = yaw_delta_init + 2*np.pi
    yaw_delta = yaw_delta_init if np.abs(yaw_delta_init) < np.abs(yaw_delta_wrap) else yaw_delta_wrap
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
    # yaw = traj_data["yaw"]
    pos = traj_data["position"]
    pos_abs =  pos - pos[0]
    if len(pos) <= 1:
        return
    yaw = [np.arctan2((pos_abs[i+1,1] - pos_abs[i,1]), (pos_abs[i+1,0] - pos_abs[i,0])) for i in range(len(pos_abs)-1)]
    yaw = yaw + [yaw[-1]]
    yaw = yaw - yaw[0]
    dataset_name = path.split('/')[-2]
    while i < total_traj_len:
        # Get traj range
        curr_traj_len = 1
        while i+curr_traj_len < total_traj_len and np.abs(get_yaw_delta(yaw[i], yaw[i+curr_traj_len])) < MIN_TURN_THRESHOLD and curr_traj_len < MAX_STEPS:
            curr_traj_len += 1
        if i+curr_traj_len >= total_traj_len:
            break
        if get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) > MIN_TURN_THRESHOLD:
            if DEBUG:
                print("Result is turn left")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[0]
            varied_language_instruction = random.choice(varied_left)
        elif get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) < -MIN_TURN_THRESHOLD:
            if DEBUG:
                print("Result is turn right")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[1]
            varied_language_instruction = random.choice(varied_right)
        elif np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1)) > STOP_THRESHOLD and np.abs(get_yaw_delta(yaw[i], yaw[i+curr_traj_len])) < MIN_FORWARD_THRESHOLD:
            if DEBUG:
                print("Result is go forward")
                print(f"Distance: {np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1))}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[2]
            varied_language_instruction = random.choice(varied_forward)
        elif np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1)) < STOP_THRESHOLD:
            if DEBUG:
                print("Result is stop")
                print(f"Distance: {np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1))}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[3]
            varied_language_instruction = random.choice(varied_stop)
        elif get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) < MIN_TURN_THRESHOLD and get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) > MIN_FORWARD_THRESHOLD:
            if DEBUG:
                print("Result is adjust left")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[4]
            varied_language_instruction = random.choice(varied_mid_left)
        elif get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) > -MIN_TURN_THRESHOLD and get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) < -MIN_FORWARD_THRESHOLD:
            if DEBUG:
                print("Result is adjust right")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[5]
            varied_language_instruction = random.choice(varied_mid_right)
        else:
            i += curr_traj_len
            continue

        curr_traj_data = {key: val[i:i+curr_traj_len+1] for key, val in traj_data.items()}
        curr_traj_data["language_instruction"] = language_instruction
        curr_traj_data["varied_language_instruction"] = varied_language_instruction
        if DEBUG:
            # Plot trajectory
            print(f"Curr trajectory yaw: {curr_traj_data['yaw']}")
            print(f"Curr traj yaw delta: {get_yaw_delta(curr_traj_data['yaw'][0], curr_traj_data['yaw'][-1])}")
            plt.plot(curr_traj_data["position"][:,0], curr_traj_data["position"][:,1])
            plt.scatter(curr_traj_data["position"][0,0], curr_traj_data["position"][0,1], c='r')
            plt.scatter(curr_traj_data["position"][-1,0], curr_traj_data["position"][-1,1], c='g')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
        
        instruction_folder = ('_').join(language_instruction.lower().split(' '))
        traj_name = f"{path.split('/')[-1]}_chunk_{chunk_idx}"
        curr_output_path = os.path.join(OUTPUT_PATH, dataset_name, instruction_folder, traj_name)
        os.makedirs(curr_output_path, exist_ok=True)
        with open(curr_output_path + f"/traj_data.pkl", "wb") as f:
            pkl.dump(curr_traj_data, f)
        with open(curr_output_path + f"/{('_').join(language_instruction.lower().split(' '))}.txt", "w") as f:
            f.write(language_instruction)
        for j in range(i, i+curr_traj_len+1):
            curr_ind = j - i
            image = iio.imread(os.path.join(path, f"{j}.jpg"))
            iio.imwrite(curr_output_path + f"/{curr_ind}.jpg", image)  
        if DEBUG:
            images = [Image.open(os.path.join(curr_output_path, f"{j-i}.jpg")) for j in range(i, i+curr_traj_len+1)]
            images[0].save("trajectory.gif", save_all=True, append_images=images[1:], duration=100, loop=1)
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






