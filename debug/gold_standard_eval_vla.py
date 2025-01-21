from PIL import Image
import os
from io import BytesIO
import argparse
import clip
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import List, Tuple, Dict, Optional
import glob 
import pickle as pkl
import requests
import base64

from data.data_utils import IMAGE_ASPECT_RATIO, VISUALIZATION_IMAGE_SIZE
from train.visualizing.action_utils import plot_trajs_and_points, plot_trajs_and_points_on_image
from train.visualizing.visualize_utils import (
    to_numpy,
    numpy_to_img,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)
from copy import copy
IMAGE_SIZE = (224, 224)

# load data_config.yaml
with open(os.path.join("/home/noam/LLLwL/lcbc/data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)

# Utility functions
def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

def transform_images_vla(pil_imgs: List[Image.Image], image_size: List[int], center_crop: bool = False):
    """Transforms a list of PIL image to a torch tensor."""
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = np.array(pil_img)
        transf_img = np.expand_dims(transf_img, axis=0)
        transf_imgs.append(transf_img)
    return np.concatenate(transf_imgs, axis=0)

def compare_output(traj_1, traj_2, gt_trajs, viz_imgs, prompt_1, prompt_2, model_name, step, fig, ax):
    dataset_name = "sacson"
    start_pos = np.array([0,0])
    goal_pos_1 = gt_trajs[0][-1]
    goal_pos_2 = gt_trajs[1][-1]
    if len(traj_1.shape) == 2:
        traj_1 = np.expand_dims(traj_1, axis=0)
    if len(traj_2.shape) == 2:
        traj_2 = np.expand_dims(traj_2, axis=0)
    for i in range(traj_1.shape[0]):
        curr_trajs = [traj_1[i], gt_trajs[0]]
        plot_trajs_and_points(
            ax[0,0], 
            curr_trajs,
            [curr_trajs[0][0], goal_pos_1], 
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
        plot_trajs_and_points_on_image(      
            ax[1,0],
            np.array(viz_imgs[0]),
            dataset_name,
            curr_trajs,
            [start_pos, goal_pos_1],
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
    for i in range(traj_2.shape[0]):
        curr_trajs = [traj_2[i], gt_trajs[1]]
        plot_trajs_and_points(
            ax[0,1], 
            curr_trajs,
            [curr_trajs[0][0], goal_pos_2], 
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
        plot_trajs_and_points_on_image(      
            ax[1,1],
            np.array(viz_imgs[1]),
            dataset_name,
            curr_trajs,
            [start_pos, goal_pos_2],
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
    # ax[0,0].set_ylim((-1, 4))
    # ax[0,0].set_xlim((-1, 4))
    # ax[0,1].set_ylim((-1, 4))
    # ax[0,1].set_xlim((-1, 4))
    ax[0,0].set_title(prompt_1)
    ax[0,1].set_title(prompt_2)
    ax[0,0].get_legend().remove()
    ax[0,1].get_legend().remove()
    prompt_1_joined = ("_").join(prompt_1.split())
    prompt_2_joined = ("_").join(prompt_2.split())
    os.makedirs(f"outputs/{model_name}", exist_ok=True)
    output_path = f"outputs/{args.model_name}/{prompt_1_joined}_vs_{prompt_2_joined}_{step}.png"
    plt.savefig(output_path)

    return output_path

def load_config(config_path):
    with open("/home/noam/LLLwL/lcbc/config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    return config


def get_vla_output(obs, prompt, server_address):
    print("Getting VLA output")
    obs_base64 = image_to_base64(Image.fromarray(obs))
    req_str = server_address + str("/gen_action")
    response = requests.post(req_str, json={'obs': obs_base64, 'prompt':prompt}, timeout=99999999)
    action = np.array(response.json()['action']).reshape(-1, 2)
    return action
        
def main(args): 

    if args.eval_path_1:
        prompt_1 = args.eval_path_1.split("/")[-1]
        traj_data_1 = np.load(os.path.join(args.eval_path_1, "traj_data.pkl"), allow_pickle=True)
        pos_1 = np.array(traj_data_1["position"])
        pos_1 = pos_1 - pos_1[0]

        smooth_pos = pos_1.copy()
        smooth_pos[np.where(np.abs(pos_1) < 1e-2)] = 0
        non_zero_idx = np.where(np.any(smooth_pos != 0, axis=1))[0][0]
        non_zero_idx = max(3, non_zero_idx)

        init_yaw = np.arctan2(pos_1[non_zero_idx, 1], pos_1[non_zero_idx, 0])
        rot_mat = np.array([[np.cos(init_yaw), -np.sin(init_yaw)], [np.sin(init_yaw), np.cos(init_yaw)]])
        pos_1 = np.dot(pos_1, rot_mat)
        new_yaw = np.arctan2(pos_1[non_zero_idx, 1], pos_1[non_zero_idx, 0])

    if args.eval_path_2:
        prompt_2 = args.eval_path_2.split("/")[-1]
        traj_data_2 = np.load(os.path.join(args.eval_path_2, "traj_data.pkl"), allow_pickle=True)
        pos_2 = np.array(traj_data_2["position"])
        pos_2 = pos_2 - pos_2[0]

        smooth_pos = pos_2.copy()
        smooth_pos[np.where(np.abs(pos_2) < 1e-2)] = 0
        non_zero_idx = np.where(np.any(smooth_pos != 0, axis=1))[0][0]
        non_zero_idx = max(3, non_zero_idx)

        init_yaw = np.arctan2(pos_2[non_zero_idx, 1], pos_2[non_zero_idx, 0])
        rot_mat = np.array([[np.cos(init_yaw), -np.sin(init_yaw)], [np.sin(init_yaw), np.cos(init_yaw)]])
        pos_2 = np.dot(pos_2, rot_mat)
        new_yaw = np.arctan2(pos_2[non_zero_idx, 1], pos_2[non_zero_idx, 0])

    num_imgs = min(len(glob.glob(os.path.join(args.eval_path_1, "*.jpg"))), len(glob.glob(os.path.join(args.eval_path_2, "*.jpg"))))
    viz_images = []
    actions_1 = np.zeros((0, 2))
    actions_2 = np.zeros((0, 2))
    fig, ax = plt.subplots(2, 2, figsize=(20,20))
    for idx in range(num_imgs - args.context_size):
        print(f"On step: {idx} of {num_imgs - args.context_size}")
        if args.eval_path_1:
            context_1 = [Image.open(os.path.join(args.eval_path_1, str(i)+".jpg")) for i in range(idx, idx + args.context_size)]
        if args.eval_path_2:
            context_2 = [Image.open(os.path.join(args.eval_path_2, str(i)+".jpg")) for i in range(idx, idx + args.context_size)]

        context_1_transf = transform_images_vla(context_1, IMAGE_SIZE).squeeze()
        context_2_transf = transform_images_vla(context_2, IMAGE_SIZE).squeeze()

        viz_1 = transform_images_vla(context_1[-1], VISUALIZATION_IMAGE_SIZE)[0] 
        viz_2 = transform_images_vla(context_2[-1], VISUALIZATION_IMAGE_SIZE)[0]

        output_1 = get_vla_output(context_1_transf, prompt_1, args.server_address)*data_config["sacson"]["metric_waypoint_spacing"]
        output_2 = get_vla_output(context_2_transf, prompt_2, args.server_address)*data_config["sacson"]["metric_waypoint_spacing"]

        summed_output_1 = np.cumsum(output_1, axis=0)[:2,:]*-1
        summed_output_2 = np.cumsum(output_2, axis=0)[:2,:]*-1

        # output_1 = summed_output_1 - summed_output_1[0]
        # output_2 = summed_output_2 - summed_output_2[0]
        actions_1 = (summed_output_1 - summed_output_1[0]) + pos_1[idx]
        actions_2 = (summed_output_2 - summed_output_2[0]) + pos_2[idx]

        print("action 1: ", output_1[0,:])
        print("action 2: ", output_2[0,:])

        # if actions_1.shape[0] == 0:
        #     actions_1 = output_1
        #     actions_2 = output_2
        #     # actions_1 = output_1[[0]]
        #     # actions_2 = output_2[[0]]

        # elif actions_1.shape[0] == 1:
        #     # delta_yaw_1 = np.arctan2(actions_1[-1,1], actions_1[-1,0])
        #     # rot_mat_1 = np.array([[np.cos(delta_yaw_1), -np.sin(delta_yaw_1)], [np.sin(delta_yaw_1), np.cos(delta_yaw_1)]])
        #     # actions_1 = np.dot(output_1, rot_mat) + actions_1[1,:]
        #     # actions_1 = output_1[0] + actions_1[1,:]
        #     actions_1 = np.vstack((actions_1, output_1[[0]] + actions_1[-1,:]))
        #     # delta_yaw_2 = np.arctan2(actions_2[-1,1], actions_2[-1,0])
        #     # rot_mat_2 = np.array([[np.cos(delta_yaw_2), -np.sin(delta_yaw_2)], [np.sin(delta_yaw_2), np.cos(delta_yaw_2)]])
        #     # actions_2 = np.dot(output_2, rot_mat_2) + actions_2[1,:]
        #     # actions_2 = output_2 + actions_2[1,:]
        #     actions_2 = np.vstack((actions_2, output_2[[0]] + actions_1[-1,:]))
        # else:
        #     # delta_yaw_1 = np.arctan2(actions_1[-1, 1] - actions_1[-2, 1] , actions_1[-1, 0] - actions_1[-2, 0])
        #     # rot_mat_1 = np.array([[np.cos(delta_yaw_1), -np.sin(delta_yaw_1)], [np.sin(delta_yaw_1), np.cos(delta_yaw_1)]])
        #     # actions_1 = np.dot(output_1, rot_mat_1) + actions_1[1,:]
        #     # actions_1 = output_1[0] + actions_1[1,:]
        #     actions_1 = np.vstack((actions_1, output_1[[0]] + actions_1[-1,:]))

        #     # delta_yaw_2 = np.arctan2(actions_2[-1, 1] - actions_2[-2, 1] , actions_2[-1, 0] - actions_2[-2, 0])
        #     # rot_mat_2 = np.array([[np.cos(delta_yaw_2), -np.sin(delta_yaw_2)], [np.sin(delta_yaw_2), np.cos(delta_yaw_2)]])
        #     # actions_2 = np.dot(output_2, rot_mat_2) + actions_2[1,:]
        #     # actions_2 = output_2[0] + actions_2[1,:]
        #     actions_2 = np.vstack((actions_2, output_2[[0]] + actions_2[-1,:]))
        
        print(actions_1)
        print(actions_2)


        viz_path = compare_output(actions_1, actions_2, [pos_1, pos_2], [viz_1, viz_2], prompt_1, prompt_2, args.model_name, idx, fig, ax)
        viz_image = Image.open(viz_path)
        viz_images.append(viz_image)
        
    viz_images[0].save(f"outputs/{args.model_name}/{prompt_1}_vs_{prompt_2}.gif", save_all=True, append_images=viz_images[1:], duration=1000, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name", default="vla-nav")
    parser.add_argument("--eval_path_1", type=str, help="path to eval data", default="gnm_version/avoid_the_person")
    parser.add_argument("--eval_path_2", type=str, help="path to eval data", default="gnm_version/move_toward_the_person")
    parser.add_argument("--context_size", type=int, help="number of samples", default=1)
    parser.add_argument("--server_address", type=str, help="server address", default="http://localhost:5000")
    args = parser.parse_args()

    main(args)

