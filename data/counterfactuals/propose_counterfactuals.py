import google.generativeai as genai
from PIL import Image
import os
import random
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from openai import OpenAI
from typing import Optional, List
import argparse
import base64
import yaml 
import shutil
import pickle as pkl
import json
from tqdm import tqdm 
import cv2
import torch
import torchvision.transforms.functional as TF

# Atomic model
from model.lelan.lnp_comp import LNPMultiModal
from model.lelan.lnp import DenseNetwork_lnp, DenseNetwork, LNP_MM
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from train.training.train_utils import replace_bn_with_gn, model_output_diffusion_eval
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Language tokenizer
from transformers import T5EncoderModel, T5Tokenizer

from data.data_utils import IMAGE_ASPECT_RATIO, VISUALIZATION_IMAGE_SIZE
from train.training.train_eval_loop import load_model

IMAGE_SIZE = (96, 96)
CAMERA_METRICS = {"camera_height" : 0.95, # meters
                "camera_x_offset" : 0.45, # distance between the center of the robot and the forward facing camera
                "camera_matrix" : {"fx": 272.547000, "fy": 266.358000, "cx": 320.000000, "cy": 220.000000},
                "dist_coeffs" : {"k1": -0.038483, "k2": -0.010456, "p1": 0.003930, "p2": -0.001007, "k3": 0.000000}}
VIZ_IMAGE_SIZE = (480, 640)  # (height, width)

# Utility functions
def pil_to_base64(img):
    img.save("temp.jpg")
    with open("temp.jpg", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def draw_trajectory(path, start, end):
    print("Drawing trajectory")
    # Load the trajectory info 
    with open(path + "/traj_data.pkl", "rb") as f:
        traj_data = pkl.load(f)
    curr_traj_data = {key: traj_data[key][start:end] for key in traj_data.keys()}
    
    # Create trajectory 
    odom = np.hstack((curr_traj_data["position"], curr_traj_data["yaw"].reshape(-1, 1)))
    odom = odom - odom[0, :]
    # project onto the image
    img = Image.open(path + f"/{start}.jpg")
    img = img.resize((VIZ_IMAGE_SIZE[1], VIZ_IMAGE_SIZE[0]))
    fig, ax = plt.subplots()
    ax.imshow(img)

    goal_img = Image.open(path + f"/{end-1}.jpg")
    
    camera_height = CAMERA_METRICS["camera_height"]
    camera_x_offset = CAMERA_METRICS["camera_x_offset"]

    fx = CAMERA_METRICS["camera_matrix"]["fx"]
    fy = CAMERA_METRICS["camera_matrix"]["fy"]
    cx = CAMERA_METRICS["camera_matrix"]["cx"]
    cy = CAMERA_METRICS["camera_matrix"]["cy"]
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    k1 = CAMERA_METRICS["dist_coeffs"]["k1"]
    k2 = CAMERA_METRICS["dist_coeffs"]["k2"]
    p1 = CAMERA_METRICS["dist_coeffs"]["p1"]
    p2 = CAMERA_METRICS["dist_coeffs"]["p2"]
    k3 = CAMERA_METRICS["dist_coeffs"]["k3"]
    dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

    xy_coords = odom[:, :2]  # (horizon, 2)
    traj_pixels = get_pos_pixels(
        xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
    )
    if len(traj_pixels.shape) == 2:
        ax.plot(
            traj_pixels[:250, 0],
            traj_pixels[:250, 1],
            color="blue",
            lw=2.5,
        )

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[1] - 0.5))
        ax.set_ylim((VIZ_IMAGE_SIZE[0] - 0.5, 0.5))
        # return the image
        plt.savefig("temp.jpg")
        out_img = Image.open("temp.jpg")
        plt.close()

        return [out_img, goal_img]
    else:
        return None

def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv

def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: Optional[bool] = False,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels

def transform_images(pil_imgs: List[Image.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
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
        transf_img = TF.to_tensor(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def get_yaw_delta(yaw_1, yaw_2):
    # Get the two cases of yaw delta
    yaw_delta_init = yaw_2 - yaw_1
    if (yaw_delta_init > 0):
        yaw_delta_wrap = yaw_delta_init - 2*np.pi
    else:
        yaw_delta_wrap = yaw_delta_init + 2*np.pi
    yaw_delta = yaw_delta_init if np.abs(yaw_delta_init) < np.abs(yaw_delta_wrap) else yaw_delta_wrap
    return yaw_delta

def discretize_traj(path, config):
    # Load base instructions
    base_instructions = config["base_instructions"]

    # Load trajectory data
    total_traj_len = len(glob.glob(path + "/*.jpg"))
    if not os.path.isdir(path):
        return
    with open(path + "/traj_data.pkl", "rb") as f:
        traj_data = pkl.load(f) 

    # Get pos and yaw data
    pos = traj_data["position"]
    pos_abs =  pos - pos[0]
    if len(pos) <= 1:
        return
    yaw = [np.arctan2((pos_abs[i+1,1] - pos_abs[i,1]), (pos_abs[i+1,0] - pos_abs[i,0])) for i in range(len(pos_abs)-1)]
    yaw = yaw + [yaw[-1]]
    yaw = yaw - yaw[0]

    # Loop through traj to get atomic actions
    i = 0
    atomic_trajs = []
    while i <= total_traj_len:
        # Get traj range
        curr_traj_len = 1
        
        # Get chunk of trajectory until turn occurs or max steps reached
        while i+curr_traj_len < total_traj_len and np.abs(get_yaw_delta(yaw[i], yaw[i+curr_traj_len])) < config["min_turn_thres"] and curr_traj_len < config["max_steps"]:
            curr_traj_len += 1
        
        # If the length of the chunk is greater than the traj, adjust the length
        if i+curr_traj_len == total_traj_len:
            curr_traj_len -= 1

        # Check if the chunk is turning right
        if get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) > config["min_turn_thres"]:
            if config["debug"]:
                print("Result is turn left")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[0]

        # Check if the chunk is turning left    
        elif get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) < -config["min_turn_thres"]:
            if config["debug"]:
                print("Result is turn right")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[1]

        # Check if the chunk is going forward
        elif np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1)) > config["stop_thres"] and np.abs(get_yaw_delta(yaw[i], yaw[i+curr_traj_len])) < config["min_turn_thres"]:
            if config["debug"]:
                print("Result is go forward")
                print(f"Distance: {np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1))}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[2]
        
        # Check if the chunk is stopping
        elif np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1)) < config["stop_thres"]:
            if config["debug"]:
                print("Result is stop")
                print(f"Distance: {np.sqrt(np.sum(np.square(pos[i+curr_traj_len,:] - pos[i,:]), axis=-1))}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[3]

        # Check if the chunk is adjusting left
        elif get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) < config["min_turn_thres"] and get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) > config["min_forward_thres"]:
            if config["debug"]:
                print("Result is adjust left")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[4]

        # Check if the chunk is adjusting right
        elif get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) > -config["min_turn_thres"] and get_yaw_delta(yaw[i], yaw[i+curr_traj_len]) < -config["min_forward_thres"]:
            if config["debug"]:
                print("Result is adjust right")
                print(f"Yaw delta: {get_yaw_delta(yaw[i], yaw[i+curr_traj_len])}")
                print(f"Yaw: {yaw[i:i+curr_traj_len+1]}")
            language_instruction = base_instructions[5]
        else:
            i += curr_traj_len
            continue
        
        # Get traj data for current chunk
        curr_traj_data = {key: val[i:i+curr_traj_len+1] for key, val in traj_data.items()}
        curr_traj_data["language_instruction"] = language_instruction
        curr_traj_data["start_idx"] = i
        curr_traj_data["end_idx"] = i + curr_traj_len
        curr_traj_data["path"] = path

        # Append to atomic trajs
        atomic_trajs.append(curr_traj_data)

        if config["debug"]:
            # Plot trajectory
            print(f"Curr trajectory yaw: {curr_traj_data['yaw']}")
            print(f"Curr traj yaw delta: {get_yaw_delta(curr_traj_data['yaw'][0], curr_traj_data['yaw'][-1])}")
            plt.plot(curr_traj_data["position"][:,0], curr_traj_data["position"][:,1])
            plt.scatter(curr_traj_data["position"][0,0], curr_traj_data["position"][0,1], c='r')
            plt.scatter(curr_traj_data["position"][-1,0], curr_traj_data["position"][-1,1], c='g')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
        
        i += curr_traj_len
        if i >= total_traj_len - 1:
            break

    return atomic_trajs

def get_model(config):
    vision_encoder = LNPMultiModal(
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        lang_encoding_size=config["lang_encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        late_fusion=config["late_fusion"],
        per_obs_film=config["per_obs_film"],
        use_film=config["use_film"],
        use_transformer=config["use_transformer"],
        )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    if config["action_head"] == "diffusion":
        action_head = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"] if not config["categorical"] else 4,
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    if config["action_head"] == "dense":
        action_head = DenseNetwork_lnp(embedding_dim=config["encoding_size"] if not config["categorical"] else 4, control_horizon=config["len_traj_pred"])
    if not config["categorical"]:
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    else:
        dist_pred_network = None
    model = LNP_MM(
        vision_encoder=vision_encoder,
        action_head=action_head,
        dist_pred_net=dist_pred_network,
        action_head_type=config["action_head"],
    )  
    return model, noise_scheduler

def t5_embed(text):
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5EncoderModel.from_pretrained("google-t5/t5-small")
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    text_features = model(tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state.mean(dim=1)
    return text_features

def model_output(model, context, prompt_embedding, prompt, noise_scheduler, model_config, device):

    output = model_output_diffusion_eval(
        model,
        noise_scheduler,
        context.clone(),
        prompt_embedding.float(),
        [prompt],
        None,
        model_config["len_traj_pred"],
        2,
        5,
        1,
        device,
        False,
        False,
    )
    return output["actions"].detach().cpu().numpy()

def main(args):

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(config["model_config"], "r") as f:
        model_config = yaml.safe_load(f)
    with open("../data_config.yaml", "r") as f:
        data_config = yaml.safe_load(f)
        config["data_config"] = data_config

    # Check of counterfactuals folder exists
    os.makedirs(config["counterfactuals_output"], exist_ok=True)

    # Load VLM 
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
    ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
    client = OpenAI(api_key=OPENAI_KEY,organization = ORGANIZATION_ID)

    # Load atomic model
    checkpoint_path = os.path.join(config["model_path"], f"{config['checkpoint']}.pth")
    latest_checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(1)) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
    model, noise_scheduler = get_model(model_config)
    load_model(model, config["model_type"], latest_checkpoint)
    model.to(args.device)
    model.eval()

    # Embed atomic actions
    action_embeddings = {}
    for action in config["base_instructions"]:
        action_embeddings[action] = t5_embed(action)

    # Select a path
    paths = glob.glob(args.data_dir + "/*")

    for path in paths:
        print(path)

        cf_folder = config["counterfactuals_output"] + f"/{path.split('/')[-1]}_cf_0"

        if os.path.exists(cf_folder):
            print("Already processed")
            continue

        # Load trajectory info
        traj_data_path = path + "/traj_data.pkl"

        # Get the original language label
        traj_data = np.load(traj_data_path, allow_pickle=True)
        if traj_data["yaw"].shape[0] < 5:
            continue
        orig_lang = [lang["traj_description"] for lang in traj_data["language_annotations"]]

        # Discretize trajectory into atomic actions
        atomic_trajs = discretize_traj(path, config)

        # Print atomic trajs
        labels = [traj["language_instruction"] for traj in atomic_trajs]
        if config["debug"]:
            print(f"Number of atomic trajs: {len(atomic_trajs)}")
            print(f"Labels: {labels}")
        # Plot the full trajectory on the image
        len_traj = len(glob.glob(path + "/*.jpg"))
        print(f"Length of trajectory: {len_traj}")
        [out_img, goal_img] = draw_trajectory(path, 0, len_traj-1)

        if args.viz:
            plt.imshow(out_img)
            plt.title("Trajectory")
            plt.axis("off")
            plt.savefig("out_viz.jpg")
        
        # Prompt the model 
        image_base64 = pil_to_base64(out_img)
        check_prompt=f"The image is the trajectory a robot took projected onto its initial observation. The actions based only on the robot odometry are {labels} and therefore will not provide information on the environment. The original instructions proposed to correspond to the trajectory based only on the robot observations are {orig_lang} and therefore will not have information grounded in the actual odometry of the robot except for what can be deduced from images. Which of the noisy original instructions makes the most sense given the actions and the observation? Additionally, provide a simple new language instruction that makes sense given the provided information. Format the response as a json with the keys 'best' and 'new'. The best field should contain a list of strings that correspond to the best original instructions. The new field should contain a list that corresponds to new instructions."
        # print(check_prompt)

        # Ask the VLM about the best and new instructions
        context = []
        context.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        })
        context.append(check_prompt)

        message = {"role": "user", "content": context}
        message_history = [message]
        response = client.chat.completions.create(
                model='gpt-4o',
                messages=message_history,
            )
        output = response.choices[0].message.content
        instruct_json = json.loads(output.lstrip("```json").rstrip("```").strip("\n").strip(" "))
        filtered_lang = [best_lang for best_lang in instruct_json["best"]] + [new_lang for new_lang in instruct_json["new"]]
        print("Filtered lang: ", filtered_lang)

        # Prompt the model to propose counterfactuals
        new_prompt=f"The image is the trajectory a robot took projected onto its initial observation. The low level actions taken by the robot are {labels} and the high level instructions that have been proposed to be associated with the trajectory are {filtered_lang}. Propose a different trajectory the robot could have taken to interact with the environment in a different way. For example, is the robot is in a hall, it can travel along the walls or in the center. Another example is that the robot could move to a specifc object in the scene. Enumerate several different alternatives. Another example is if the robot Only propose short horizon alternatives and provide specific information about the task. Give the previous low level action and its index in the low level actions list from which the trajectory should take an alternative path and then low level action, from the list: ['Turn left', 'Turn right', 'Go forward', 'Stop', 'Adjust left', 'Adjust right'] which performs the alternative path. Your output should be in the form of a list of json objects with a field for the trajectory and a field for reasoning. For example, if the input low level actions are ['Go forward', 'Go forward', 'Turn left'], the original instruction was 'Move towards the door on the left' then a potential output could be : '['prev_action' : ('Go forward', 1), 'proposed_action' : 'Turn right', 'new_instruction' : ' Move away from the door on the left' 'reasoning': 'The robot could try instead moving away from the door on the left to explore the room further. This would be a good alternative to the original instruction.'"

        # Ask the VLM about the best and new instructions
        context = []
        context.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        })
        context.append(new_prompt)

        message = {"role": "user", "content": context}
        message_history = [message]
        tries = 0
        while tries < 10:
            try:
                response = client.chat.completions.create(
                        model='gpt-4o',
                        messages=message_history,
                    )
                output = response.choices[0].message.content
                cf_json = json.loads(output.lstrip("```json").rstrip("```"))
                check_keys = ["prev_action", "proposed_action", "new_instruction", "reasoning"]
                for cf in cf_json:
                    for key in check_keys:
                        assert key in cf.keys()
                        if key == "prev_action":
                            assert cf["prev_action"][0] in labels
                            assert cf["proposed_action"] in config["base_instructions"]
                            assert cf["prev_action"][1] < len(atomic_trajs)
                            assert len(cf["prev_action"]) == 2
                break
            except:
                tries += 1
                pass

        # Next step, annotate with rollout from model 
        print("Num of counterfactuals: ", len(cf_json))
        for cf_idx, cf in enumerate(cf_json):
            action, idx = cf["prev_action"]
            proposed_action = cf["proposed_action"]
            new_instruction = cf["new_instruction"]
            
            # Load in the context associated with the action
            print(f"Action: {action}, Index: {idx}")
            prev_atomic = atomic_trajs[idx]
            if prev_atomic["end_idx"] < model_config["context_size"] - 1 or (prev_atomic["end_idx"] - model_config["context_size"] - 1) < 0:
                print("Not enough context")
                continue
            print(f"End idx: {prev_atomic['end_idx']}")
            print(f"End idx: {prev_atomic['end_idx'] - model_config['context_size'] - 1}")
            context = [Image.open(path + f"/{i}.jpg") for i in range(prev_atomic["end_idx"] - model_config["context_size"] - 1, prev_atomic["end_idx"])]
            context = transform_images(context, IMAGE_SIZE).to(args.device)

            viz_context = transform_images(Image.open(path + f"/{prev_atomic['end_idx']}.jpg"), VIZ_IMAGE_SIZE)

            lang_embed = action_embeddings[proposed_action].to(args.device)

            # Get the model rollout 
            rollout = model_output(model, context, lang_embed, action, noise_scheduler, model_config, args.device)
            rollout = rollout - rollout[:,[0],:]

            # transform into the frame of the original trajectory
            last_pos = traj_data["position"][prev_atomic["end_idx"]-1]
            last_yaw = float(traj_data["yaw"][prev_atomic["end_idx"]-1])
            rot_mat = np.array([[np.cos(last_yaw), -np.sin(last_yaw)], [np.sin(last_yaw), np.cos(last_yaw)]]).reshape(-1, 2, 2)
            rollout_rotated = rot_mat@np.transpose(rollout, (0, 2, 1)) # rotate
            rollout_rotated = np.transpose(rollout_rotated, (0, 2, 1))

            # unnorm the rollout
            dataset_name = path.split("/")[-2]
            waypoint_spacing = data_config[dataset_name]["metric_waypoint_spacing"]
            rollout = rollout * waypoint_spacing
            rollout = rollout_rotated + np.expand_dims(last_pos, 0) # translate


            # Combine the part of the old traj with the new traj
            old_traj = traj_data["position"][:prev_atomic["end_idx"]+1].copy()
            cf_trajs = []
            for i in range(rollout.shape[0]):
                cf_traj = np.concatenate([old_traj, rollout[i,:,:]], axis=0)
                cf_trajs.append(cf_traj)

            # Plot both trajs
            # plt.close()
            # fig, ax = plt.subplots(1,2)
            # for i in range(len(cf_trajs)):
            #     ax[0].plot(cf_trajs[i][:,0], cf_trajs[i][:,1])
            # ax[0].plot(traj_data["position"][:,0], traj_data["position"][:,1], label="Original", color="red")
            # ax[0].scatter(traj_data["position"][0,0], traj_data["position"][0,1], c='g')
            # ax[0].scatter(traj_data["position"][-1,0], traj_data["position"][-1,1], c='b')
            # ax[0].set_title(f"Counterfactual Trajectory: {proposed_action}")

            # ax[1].imshow(viz_context.squeeze().permute(1,2,0))
            # ax[1].set_title("Context")
            # plot_name = ('_').join(cf['new_instruction'].lower().split(' '))
            # if os.path.exists(f"cf_traj_{plot_name}.jpg"):
            #     os.remove(f"cf_traj_{plot_name}.jpg")
            # plt.savefig(f"cf_traj_{plot_name}.jpg")

            # Save the counterfactuals to a new folder
            cf_folder = config["counterfactuals_output"] + f"/{path.split('/')[-1]}_cf_{cf_idx}"
            print(cf_folder)
            if not os.path.exists(cf_folder):
                os.makedirs(cf_folder)
            
            rollout = rollout[0,:,:]
            cf_traj_data = {}
            cf_traj_data["position"] = cf_traj
            cf_traj_data["yaw"] = [np.arctan2((cf_traj[i+1,1] - cf_traj[i,1]), (cf_traj[i+1,0] - cf_traj[i,0])) for i in range(len(cf_traj)-1)]
            cf_traj_data["language_annotations"] = [{"traj_description": new_instruction}]

            with open(cf_folder + "/traj_data_filtered.pkl", "wb") as f:
                pkl.dump(cf_traj_data, f)
            
            for image_idx in range(old_traj.shape[0]):
                shutil.copy(path + f"/{image_idx}.jpg", cf_folder + f"/{image_idx}.jpg")

        # Save the filtered language instructions to a copied traj_data
        traj_data_filtered = traj_data.copy()
        traj_data_filtered["language_annotations"] = [{"traj_description": lang} for lang in filtered_lang]
        traj_data_filtered_path = path + "/traj_data_filtered.pkl"
        with open(traj_data_filtered_path, "wb") as f:
            pkl.dump(traj_data_filtered, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/counterfactuals")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--config_path", type=str, default="counterfactuals.yaml")
    args = parser.parse_args()

    device = "cuda:1" if torch.cuda.is_available() else None
    args.device = device
    
    main(args)


    

