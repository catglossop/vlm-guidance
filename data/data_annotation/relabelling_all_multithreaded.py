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
from multiprocessing import Pool, Lock
from tqdm_multiprocess import TqdmMultiProcessPool

# OPTIONAL (FOR VISUALIZATION ONLY)
CAMERA_METRICS = {"camera_height" : 0.95, # meters
                "camera_x_offset" : 0.45, # distance between the center of the robot and the forward facing camera
                "camera_matrix" : {"fx": 272.547000, "fy": 266.358000, "cx": 320.000000, "cy": 220.000000},
                "dist_coeffs" : {"k1": -0.038483, "k2": -0.010456, "p1": 0.003930, "p2": -0.001007, "k3": 0.000000}}
VIZ_IMAGE_SIZE = (480, 640)  # (height, width)
TRANSITION_TO_MUTLIPROCESS = True
lock = Lock()
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

def generate_trajectory(imgs):

    # perform visual odometry to get odom in frame of init robot pose 
    pose = np.zeros((3, 1))

    # initialize the feature tracker 
    pass

# Relabelling functions
def relabel_traj_gpt(images_64, prompt, client, in_context_images_64=None, in_context_text=None, model_name="gpt-4o", actions=None):
    message_history = []
    context_no_images = []
    if in_context_images_64 is not None: 
        context = [in_context_text]  
        context_no_images = [in_context_text]
        for image_base64 in in_context_images_64:
            context.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
            })
    for i, p in enumerate(prompt.keys()):
        context = []
        if actions is not None and i == 0:
            context.append(prompt[p] + (np.array_str(actions)))   
            context_no_images.append(prompt[p] + (np.array_str(actions)))  
        else:
            context.append(prompt[p])   
            context_no_images.append(prompt[p])
        if i == 0:
            for image_base64 in images_64:
                context.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    },
                })
        message = {"role": "user", "content": context}
        message_history.append(message)
        response = client.chat.completions.create(
            model=model_name,
            messages=message_history,
        )
        assistant_message = response.choices[0].message
        message_history.append(assistant_message)
        context_no_images.append(response.choices[0].message.content)
    label = response.choices[0].message.content
    return label, context_no_images

def relabel_traj_gpt_hierarchical(images_64, prompt, client, in_context_images_64=None, in_context_text=None, model_name="gpt-4o", actions=None, debug=False):
    message_history = []
    context_no_images = []
    image_descriptions = []

    if in_context_images_64 is not None: 
        context = [in_context_text]  
        context_no_images = [in_context_text]
        for image_base64 in in_context_images_64:
            context.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
            })

    for i, p in enumerate(prompt.keys()):
        context = []
        if debug:
            print("Prompt key: ", p)
        if i == 0:  
            context_no_images.append(prompt[p])

            message = {"role": "user", "content": prompt[p]}
            message_history.append(message)
            response = client.chat.completions.create(
                model=model_name,
                messages=message_history,
            )
            assistant_message = response.choices[0].message
            message_history.append(assistant_message)
            context_no_images.append(response.choices[0].message.content)
        elif i == 1:
            for j, image_base64 in enumerate(images_64):
                context = []
                context.append(prompt[p])
                context.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    },
                })
                
                message = {"role": "user", "content": context}
                message_history.append(message)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=message_history,
                )
                image_descriptions.append(response.choices[0].message.content)
                if debug:
                    print(f"RESPONSE TO IMAGE DESCRIP {j}: ", response.choices[0].message.content)
                assistant_message = response.choices[0].message
                message_history.append(assistant_message)
                context_no_images.append(response.choices[0].message.content)
        elif i == 2:
            str_image_descriptions = "[" + "', '".join(image_descriptions) + "]"
            context.append(prompt[p] + str_image_descriptions)
            message = {"role": "user", "content": context}
            message_history.append(message)
            response = client.chat.completions.create(
                model=model_name,
                messages=message_history,
            )
            assistant_message = response.choices[0].message
            if debug:
                print("RESPONSE TO FINAL CONTEXT: ", response.choices[0].message.content)
            message_history.append(assistant_message)
            context_no_images.append(response.choices[0].message.content)
    label = response.choices[0].message.content
    return label, context_no_images

def relabel_traj_gemini(images, prompt, client, in_context_images=None, in_context_text=None):
    context = []
    if in_context_images:
        context = [in_context_text]
        for img in in_context_images:
            context.append(img)
    context.append(prompt)
    for img in images:
        context.append(img)
    response = client.generate_content(context)
    label = response.text
    return label, context

def relabel_traj_prismatic(images, prompt, client, in_context_images=None, in_context_text=None):
    pass

def add_annotation(img, idx):
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    # Add annotation
    ax.text(img.size[0] - 10, 10, str(idx), fontsize=20, color="red")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim((0.5, img.size[0] - 0.5))
    ax.set_ylim((img.size[1] - 0.5, 0.5))
    plt.savefig(f"temp.jpg")
    out_img = Image.open("temp.jpg")
    plt.imshow(out_img)
    plt.show()
    return out_img
# Main function
def main(args):
    # Get model
    model = args.model
    if model == "gpt":
        OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
        ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
        client = OpenAI(api_key=OPENAI_KEY,organization = ORGANIZATION_ID)
    elif model == "gemini":
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        client = genai.GenerativeModel(model_name="gemini-1.5-flash")
    elif model == "prismatic":
        raise NotImplementedError("Prismatic is not yet supported")
    else: 
        raise ValueError("Invalid model name")
    print(f"Using model: {model} with name: {args.model_name}")

    # Get config
    period = args.period
    annotation_type = args.annotation_type
    dataset = args.dataset
    print(f"Using dataset: {dataset}")
    output = args.output
    if args.debug: 
        output = output + "_debug"
    if args.overwrite:
        shutil.rmtree(output)
        os.makedirs(output)
    else:
        os.makedirs(output, exist_ok=True)
    print(f"Outputting dataset to: {output}")
    prompt = json.load(open(args.prompt, "r"))

    # Check if using in-context examples, if so load them (for now, just one in-context example)
    in_context_images = None
    in_context_text = None
    in_context_images_base64 = None
    if args.in_context:
        print(f"Using in context: {args.in_context}")
        with open(args.in_context, "r") as in_context:
            try: 
                in_context_example = yaml.safe_load(args.in_context, "r")
            except:
                raise ValueError("Invalid in context file")
        in_context_images = [Image.open(in_context_example["path"][i]) for i in range(len(in_context_example["path"]))]
        in_context_text = in_context_example["text"]
        
        if model == "gpt":
            # convert images to base64
            in_context_images_base64 = [pil_to_base64(img) for img in in_context_images]

    # Get paths 
    # select subset of paths based on location 
    if dataset.split("/")[-1] == "sacson":
        bww1_paths = glob.glob(dataset + "/*bww1*")
        print(f"Number of bww1 paths: {len(bww1_paths)}")
        bww2_paths = glob.glob(dataset + "/*bww2*")
        print(f"Number of bww2 paths: {len(bww2_paths)}")
        bww8_paths = glob.glob(dataset + "/*bww8*")
        print(f"Number of bww8 paths: {len(bww8_paths)}")
        soda_paths = glob.glob(dataset + "/*soda3*")
        print(f"Number of soda paths: {len(soda_paths)}")
        cory_paths = glob.glob(dataset + "/*cory1*")
        print(f"Number of cory paths: {len(cory_paths)}")
        paths = bww1_paths[:100] + bww2_paths[:100] + bww8_paths[:100] + soda_paths[:100] + cory_paths[:100]
        paths = glob.glob(dataset + "/*")
    else:
        paths = glob.glob(dataset + "/*")

    print(f"Number of paths: {len(paths)}")

    if args.debug: 
        paths = random.choices(paths, k=10)
    current_state = None
    starting_from_save = False
    
    if os.path.exists(os.path.join(output, "current_state/current_state.pkl")) and TRANSITION_TO_MUTLIPROCESS:
        print("Resuming from save")
        current_state = np.load(os.path.join(output, "current_state/current_state.pkl"), allow_pickle=True)
        starting_from_save = True
        completed_paths = os.listdir(output)
        completed_paths = [p.split("_chunk_")[0] for p in completed_paths if "chunk" in p]
        completed_set = set(completed_paths)
        path_keys = [p.split("/")[-1] for p in paths]
        path_keys = set(path_keys)
        remaining_path_idxs = [i for i, p in enumerate(paths) if p.split("/")[-1] not in completed_set]
        paths = [paths[i] for i in remaining_path_idxs]
        print("Number of paths remaining: ", len(paths))
        traj_list = os.listdir(output) 
        traj_list.remove("current_state")
        print("Current state: ", current_state)

    # Process each path
    path_shards = np.array_split(paths, args.num_processes)

    tasks = [(label_trajectories, (i, path_shards[i], output, annotation_type, model, prompt, in_context_images, in_context_images_base64, in_context_text, period, args, current_state, starting_from_save)) for i in range(args.num_processes)]
    pool = TqdmMultiProcessPool(args.num_processes)
    print("Starting multiprocessing")
    with tqdm(total=len(tasks), 
              dynamic_ncols=True,
              position=0,
              desc="Total progress"
    ) as pbar:
        pool.map(pbar, tasks, lambda _: None, lambda _: None)

def label_trajectories(thread_num, paths, output, annotation_type, model, prompt, in_context_images, in_context_images_base64, in_context_text, period, args, current_state=None, starting_from_save=False, tqdm_func=None, global_tqdm=None):
    if model == "gpt":
        OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
        ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
        client = OpenAI(api_key=OPENAI_KEY,organization = ORGANIZATION_ID)
    elif model == "gemini":
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        client = genai.GenerativeModel(model_name="gemini-1.5-flash")
    elif model == "prismatic":
        raise NotImplementedError("Prismatic is not yet supported")
    else: 
        raise ValueError("Invalid model name")
    print(f"In thread {thread_num}")
    # if os.path.exists(os.path.join(output, f"current_state/current_state_{thread_num}.pkl")):
    #     print("Resuming from save")
    #     current_state = np.load(os.path.join(output, f"current_state/current_state_{thread_num}.pkl"), allow_pickle=True)
    #     starting_from_save = True
    #     paths = paths[paths.index(current_state["trajectory"]):]
    #     print("Number of paths remaining: ", len(paths))
    #     traj_list = os.listdir(output) 
    #     traj_list.remove("current_state")
    #     print("Current state: ", current_state)

    traj_lens = np.arange(args.min_traj_len, args.max_traj_len)
    for path in paths:
        chunk_idx = 0
        # print(f"Processing path: {path}")
        total_traj_len = len(glob.glob(path + "/*.jpg"))
        with open(path + "/traj_data.pkl", "rb") as f:
            traj_data = pkl.load(f) 
        if current_state is not None and starting_from_save:
            i = current_state["start"]
        else:
            i = 0
        if len(traj_data["yaw"]) < 2:
            continue
        while i < total_traj_len:
            traj_len = np.random.choice(traj_lens)
            # Get traj range
            if current_state is not None and starting_from_save:
                start = current_state["start"]
                end = current_state["end"]
                starting_from_save = False
                traj_len = end - start
            else:
                start = i
                end = min(i + traj_len, total_traj_len)
                traj_len = end - start
            # print(f"Trajectory length {traj_len} starting at {start} ending at {end} in trajectory of length {total_traj_len}")

            # Get curr traj data
            curr_traj_data = {}
            for key in traj_data.keys():
                curr_traj_data[key] = traj_data[key][start:end]
            
            if args.use_actions: 
                actions = np.hstack((curr_traj_data["position"], curr_traj_data["yaw"].reshape(-1, 1)))
                actions = actions - actions[0, :]
            
            # Get and annotate images
            if annotation_type == "drawn":
                imgs = draw_trajectory(path, start, end)
            elif annotation_type == "sampled":
                imgs = [Image.open(path + f"/{k}.jpg") for k in range(start, end, period)]
                if args.annotate:
                    temp_imgs = imgs
                    imgs = []
                    for j, img in enumerate(temp_imgs):
                        img = add_annotation(img, j)
                        imgs.append(img)
                if model == "prismatic":
                    raise ValueError("Prismatic cannot accept a sampled trajectory")
            else:
                raise ValueError("Invalid annotation type")
            
            # Convert images to base64 if using GPT-4
            if model == "gpt" and imgs is not None:
                imgs_base64 = [pil_to_base64(img) for img in imgs]
            
            # Relabel the trajectory
            current_state = {"trajectory": path, "start": start, "end": end}
            with lock:
                with open(os.path.join(output, f"current_state_{thread_num}.pkl"), "wb") as f:
                    pkl.dump(current_state, f)

            label = None
            context = None
            if model == "gpt":
                if args.prompt == "prompt_hierarchical.json" or args.prompt == "prompt_hierarchical_templated.json":
                    max_tries = 2
                    tries = 0
                    instruction_json = None
                    while tries < max_tries:
                        try:
                            label, context = relabel_traj_gpt_hierarchical(imgs_base64, prompt, client, in_context_images_base64, in_context_text, args.model_name, actions=None, debug = args.debug)
                            instruction_json = json.loads(label.lstrip("```json").rstrip("```").strip("\n").strip(" "))
                            break
                        except:
                            # print(f"Try {tries} of {max_tries}. Retrying...")
                            tries += 1
                            continue
                    assert instruction_json is not None, "Failed to get instruction json"
                else:
                    if imgs is not None: 
                        if args.use_actions:
                            label, context = relabel_traj_gpt(imgs_base64, prompt, client, in_context_images_base64, in_context_text, args.model_name, actions=actions)
                        else:
                            label, context = relabel_traj_gpt(imgs_base64, prompt, client, in_context_images_base64, in_context_text, args.model_name, actions=None)
                
            elif model == "gemini":
                label, context = relabel_traj_gemini(imgs, prompt, client, in_context_images, in_context_text)

            elif model == "prismatic":
                raise NotImplementedError("Prismatic is not yet supported")
            else:
                raise ValueError("Invalid model name")
            if label is None: 
                label = "No label"
                context = ["No context"]
            
            instructions = []
            for idx, inst in enumerate(instruction_json["instructions"]):
                instruction = {"traj_description": inst}
                instructions.append(instruction)
            curr_traj_data["language_annotations"] = instructions

            # Save the output
            traj_output_dir = os.path.join(output, f"{path.split('/')[-1]}_chunk_{chunk_idx}_start_{start}_end_{end}")
            print("saving to ", traj_output_dir)
            os.makedirs(traj_output_dir, exist_ok=True)
            for m in range(start, end):
                shutil.copyfile(path + f"/{m}.jpg", os.path.join(traj_output_dir, f"{m-start}.jpg"))
            with lock:
                pkl.dump(curr_traj_data, open(os.path.join(traj_output_dir, "traj_data.pkl"), "wb"))
            with open(os.path.join(traj_output_dir, "label.txt"), "w") as f:
                f.write(label)
            
            # If viz, visualize the trajectory and save the context
            if args.viz and imgs is not None:
                print("Visualizing trajectory")
                if annotation_type == "sampled": 
                    imgs[0].save(os.path.join(traj_output_dir, f"traj_viz_{chunk_idx}.gif"), save_all=True, append_images=imgs[1:], duration=500, loop=0)
                elif annotation_type == "drawn":
                    imgs.save(os.path.join(traj_output_dir, f"traj_viz_{chunk_idx}.jpg"))
                with open(os.path.join(traj_output_dir, "context.txt"), "w") as f:
                    for c in context:
                        f.write(f"{c}\n")
            chunk_idx += 1
            i += traj_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt", help="gpt, gemini, prismatic, all")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of specific model or path to model")
    parser.add_argument("--dataset", type=str, default="/home/noam/LLLwL/datasets/gnm_dataset/sacson", help="Path to dataset")
    parser.add_argument("--min_traj_len", type=int, default=15, help="Length of trajectory")
    parser.add_argument("--max_traj_len", type=int, default=25, help="Length of trajectory")
    parser.add_argument("--period", type=int, default=3, help="Period of trajectory (sampling rate)")
    parser.add_argument("--prompt", type=str, default="/home/noam/LLLwL/gemini/prompt.json", help="Path to prompt json file")
    parser.add_argument("--in_context", type=str, default=None, help="Path to in context yaml file with image paths and text")
    parser.add_argument("--output", type=str, default="go_stanford_cropped_labelled", help="Path to output directory")
    parser.add_argument("--annotation_type", type=str, default="sampled", help="Either drawn trajectory of sampled trajectory (choices: 'drawn', 'sampled')")
    parser.add_argument("--viz", action="store_true", help="Visualize the trajectory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory")
    parser.add_argument("--use_actions", action="store_true", help="Use actions to generate the trajectory")
    parser.add_argument("--annotate", action="store_true", help="Annotate the images with frame number")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes")
    
    args = parser.parse_args()
    args.overwrite = False
    main(args)