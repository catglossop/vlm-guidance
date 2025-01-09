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

CAMERA_METRICS = {"camera_height" : 0.95, # meters
                "camera_x_offset" : 0.45, # distance between the center of the robot and the forward facing camera
                "camera_matrix" : {"fx": 272.547000, "fy": 266.358000, "cx": 320.000000, "cy": 220.000000},
                "dist_coeffs" : {"k1": -0.038483, "k2": -0.010456, "p1": 0.003930, "p2": -0.001007, "k3": 0.000000}}

# CAMERA_METRICS = {"camera_height" : 0.60, # meters
#                 "camera_x_offset" : 0.10, # distance between the center of the robot and the forward facing camera
#                 "camera_matrix" : {"fx": 68.13675, "fy": 66.5895, "cx": 80, "cy": 55},
#                 "dist_coeffs" : {"k1": -0.038483, "k2": -0.010456, "p1": 0.003930, "p2": -0.001007, "k3": 0.000000}}

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
    breakpoint()
    return pixels

def load_model():

def main(args):

    # Select an image
    paths = glob.glob(args.data_dir + "/*")
    path = random.choice(paths)
    print(path)

    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
    ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
    client = OpenAI(api_key=OPENAI_KEY,organization = ORGANIZATION_ID)

    # plot the trajectory 
    len_traj = len(glob.glob(path + "/*.jpg"))
    end_len = min(10, len_traj)
    print(f"Length of trajectory: {len_traj}")
    [out_img, goal_img] = draw_trajectory(path, 0, end_len)

    if args.viz:
        plt.imshow(out_img)
        plt.title("Trajectory")
        plt.axis("off")
        plt.savefig("out_viz.jpg")
    
    # Prompt the model 
    image_base64 = pil_to_base64(out_img)
    prompt ="The image is the trajectory a robot took projected onto its initial observation. Propose a different trajectory the robot could have taken to interact with the environment in a different way. For example, is the robot is in a hall, it can travel along the walls or in the center. Another example is that the robot could move to a specifc object in the scene. Enumerate several different alternatives. Only propose short horizon alternatives and provide specific information about the task. Format the trajectories as a sequences of commands from the list: ['Turn left', 'Turn right', 'Move forward', 'Stop'] paired with an amount, in degrees or meters. Your output should be in the form of a list of json objects with a field for the trajectory and a field for reasoning. For example: [{'trajectory': [('Turn left', 90) ('Move forward', 1m), ('Turn right', 90)], 'reasoning': 'The robot should turn left and move closer to the white wall and then turn right to follow along it.'}]"

    context = []
    context.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    })
    context.append(prompt)

    message = {"role": "user", "content": context}
    message_history = [message]
    response = client.chat.completions.create(
            model='gpt-4o',
            messages=message_history,
        )
    print(response.choices[0].message.content)
    breakpoint()

    # Next step, annotate with rollout from model 



    # Check the possible outputs and select the best option

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/counterfactuals")
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()
    main(args)


    

