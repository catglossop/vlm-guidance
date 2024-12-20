import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from collections import OrderedDict
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import matplotlib.pyplot as plt
import yaml
import threading
import random
import io
import base64
import requests
import tensorflow as tf
import copy
import json
from datetime import datetime

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String
from nav_msgs.msg import Odometry
# from irobot_create_msgs.msg import DockStatus
from rclpy.action import ActionClient
# from irobot_create_msgs.action import Undock
from cv_bridge import CvBridge

from deployment.utils import to_numpy, transform_images, load_model
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
from difflib import SequenceMatcher

# UTILS
from deployment.topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC, 
                        REACHED_GOAL_TOPIC, 
                        ANNOTATED_IMAGE_TOPIC)

from train.visualizing.visualize_utils import (BLUE, 
                                                GREEN, 
                                                RED, 
                                                CYAN, 
                                                MAGENTA,
                                                YELLOW)
from train.visualizing.action_utils import plot_trajs_and_points_on_image

# CONSTANTS
TOPOMAP_IMAGES_DIR = "topomaps/images"
ROBOT_CONFIG_PATH ="../../config/robot.yaml"
MODEL_CONFIG_PATH = "../../config/models.yaml"
DATA_CONFIG = "../../data/data_config.yaml"
PRIMITIVES = ["Turn left", "Turn right", "Go straight", "Stop"]
DEBUG = False
HARDCODED = False
SINGLE_STEP = True
POS_THRESHOLD = 0.2 
YAW_THRESHOLD = np.pi/6

class LowLevelPolicy(Node): 

    def __init__(self, 
                args
                ):
        super().__init__('low_level_policy')
        self.args = args
        self.hl_prompt = args.prompt
        self.context_queue = []
        self.context_size = None
        self.server_ip = args.ip
        self.SERVER_ADDRESS = f"http://{self.server_ip}:5001"
        self.subgoal_timeout = 15 # This is partially for debugging
        self.wait_for_reset = False
        self.image_aspect_ratio = (4/3)
        self.starting_traj = True
        self.dists = None
        self.traj_duration = 0
        self.bridge = CvBridge()
        self.reached_dist = 10
        self.vlm_plan = None
        self.traj_yaws = []
        self.traj_pos = []
        self.primitive_matches = {}
        self.traj_idx = 0
        now = datetime.now() 
        self.date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

        # Set planning values
        self.ll_success = False
        self.ll_timeout = 20
        self.ll_steps = 0
        self.hl_success = False
        self.colors = [BLUE, GREEN, RED, CYAN, MAGENTA, YELLOW]
        self.traj_color_map = {'blue' : 0, 'green' : 1, 'red' : 2, 'cyan' : 3, 'magenta' : 4, 'yellow' : 5}
        self.step = 0
        self.plan_freq = 1
        self.ll_prompt = "GEN_NEW_PLAN"

        # Load the model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.load_model_from_config(MODEL_CONFIG_PATH)

        # Load the config
        self.load_config(ROBOT_CONFIG_PATH)

        # Load data config
        self.load_data_config()

        os.makedirs("/home/create/hi_learn_results/primitives", exist_ok=True)

        self.irobot_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
            )
        self.callback_group = ReentrantCallbackGroup()

        # SUBSCRIBERS 
        self.annotated_img_pub = self.create_publisher(
                                        Image, 
                                        ANNOTATED_IMAGE_TOPIC, 
                                        1) 
        self.image_msg = Image()
        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10, 
            callback_group=self.callback_group)
        self.subgoal_image = None
        self.odom_msg = Odometry()
        self.current_pos = None
        self.current_yaw = None
        self.odom_sub = self.create_subscription(
            Odometry, 
            "/odom",
            self.odom_callback,
            self.irobot_qos_profile)

        # PUBLISHERS
        self.reached_goal = False
        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, 
            SAMPLED_ACTIONS_TOPIC, 
            1)
        self.waypoint_msg = Float32MultiArray()
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, 
            WAYPOINT_TOPIC, 
            1) 
        self.subgoal_msg = Image()
        self.subgoal_pub = self.create_publisher(
            Image, 
            "hierarchical_learning/subgoal", 
            1) 
        
        # TIMERS
        self.timer_period = 1/self.RATE  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=self.callback_group)
        print("Setup complete")
    
    # Utils
    def image_to_base64(self, image):
        buffer = io.BytesIO()
        # Convert the image to RGB mode if it's in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str
    
    def unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data

    def get_delta(self, actions):
        # append zeros to first action
        ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
        delta = ex_actions[:,1:] - ex_actions[:,:-1]
        return delta

    def get_action(self):
        # diffusion_output: (B, 2*T+1, 1)
        # return: (B, T-1)
        ndeltas = self.naction
        ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
        ndeltas = to_numpy(ndeltas)
        ndeltas = self.unnormalize_data(ndeltas, self.ACTION_STATS)
        actions = np.cumsum(ndeltas, axis=1)
        return torch.from_numpy(actions).to(self.device)

    def transform_image_to_string(self, pil_img, image_size, center_crop=False):
        """Transforms a PIL image to a torch tensor."""
        w, h = pil_img.size
        if center_crop: 
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * self.image_aspect_ratio)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / self.image_aspect_ratio), w))
        pil_img = pil_img.resize(image_size)
        image_bytes_io = io.BytesIO()
        PILImage.fromarray(np.array(self.obs)).save(image_bytes_io, format = 'JPEG')
        image_bytes = tf.constant(image_bytes_io.getvalue(), dtype = tf.string)
        return image_bytes

    # Loading 
    def load_config(self, robot_config_path):
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.MAX_V = 0.4
        self.MAX_W = 0.2
        self.VEL_TOPIC = "/task_vel"
        self.DT = 1/robot_config["frame_rate"]
        self.RATE = robot_config["frame_rate"]
        self.EPS = 1e-8
    
    def load_model_from_config(self, model_paths_config):
        # Load configs
        with open(model_paths_config, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[self.args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"] 

        # Load model weights
        self.ckpth_path = model_paths[self.args.model]["ckpt_path"]
        if os.path.exists(self.ckpth_path):
            print(f"Loading model from {self.ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {self.ckpth_path}")
        self.model = load_model(
            self.ckpth_path,
            self.model_params,
            self.device,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
    def load_data_config(self):
        # LOAD DATA CONFIG
        with open(os.path.join(os.path.dirname(__file__), DATA_CONFIG), "r") as f:
            data_config = yaml.safe_load(f)
        # POPULATE ACTION STATS
        self.ACTION_STATS = {}
        for key in data_config['action_stats']:
            self.ACTION_STATS[key] = np.array(data_config['action_stats'][key])

    # TODO: add subscription to VLM planner to get the goal
    def get_traj_from_server(self, image: PILImage.Image) -> dict:
        # Plot the actions on the image
        print("Current LL step: ", self.ll_steps)
        image = image.convert("RGB")
        fig, ax = plt.subplots()
        annotated_image = plot_trajs_and_points_on_image(ax, 
                                        image.resize((640, 480), PILImage.Resampling.NEAREST), 
                                        "recon", self.naction, 
                                        [], self.colors, 
                                        [])
        fig.savefig("annotated_img.jpg")
        ax.set_axis_off()
        plt.tight_layout()
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='jpg')
        image = PILImage.open(img_buf)
        image_np = np.array(image).astype(np.uint8)
        annotated_msg = self.bridge.cv2_to_imgmsg(image_np, "passthrough")
        self.annotated_img_pub.publish(annotated_msg)
        image_base64 = self.image_to_base64(image)

        if self.ll_success or self.ll_steps > self.ll_timeout:
            self.ll_steps = 0
            self.ll_prompt = self.vlm_plan.pop(0)
        if self.vlm_plan is None or len(self.vlm_plan) == 0:
            assert self.ll_prompt == "GEN_NEW_PLAN"
            print("Requesting VLM plan")
            response = requests.post(self.SERVER_ADDRESS + str("/gen_plan"), json={'actions': image_base64, 'hl_prompt':self.hl_prompt}, timeout=99999999)
            res = json.loads(response.json()["plan"].lstrip("```json").rstrip("```").strip("\n").strip(" "))
            self.vlm_plan = res['plan']
            self.hl_success = res["task_success"]
            self.reasoning = res['reason']
        print("--------------------------------------------------")
        print(f"PLAN: {self.vlm_plan}")
        print(f"HL SUCCESS: {self.hl_success}")
        print(f"Reason: {self.reasoning}")
        print("--------------------------------------------------")
        if self.hl_success: 
            print("TRAJECTORY SUCCESSFUL")
            return True
        
        if type(self.ll_prompt) == list:
            self.ll_prompt = self.ll_prompt[0]
        
        response = requests.post(self.SERVER_ADDRESS + str("/verify_action"), json={'actions': image_base64, 'll_prompt':self.ll_prompt, 'hl_prompt':self.hl_prompt}, timeout=99999999)
        res =  json.loads(response.json()["traj"].lstrip("```json").rstrip("```").strip("\n").strip(" "))
        trajectory = res['trajectory'].lower()
        if trajectory == "none":
            trajectory = "red"
        self.ll_success = res["task_success"]
        try: 
            reasoning = res['reason']
        except: 
            reasoning = res['reasoning']
        print("--------------------------------------------------")
        print(f"LL prompt: {self.ll_prompt}")
        print(f"Trajectory: {trajectory}")
        print(f"Task success: {self.ll_success}")
        print(f"Reason: {reasoning}")
        print("--------------------------------------------------")

        self.chosen_action = self.naction[self.traj_color_map[trajectory]]
        self.ll_steps += 1
        return False 

    def odom_callback(self, odom_msg: Odometry):
        """Callback function for the odometry subscriber"""
        self.current_yaw = odom_msg.pose.pose.orientation.z
        self.current_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        
    def image_callback(self, msg):
        self.image_msg = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.image_msg = PILImage.fromarray(self.image_msg)
        self.obs = self.image_msg
        self.fake_goal = torch.randn((1, 3, self.model_params["image_size"][0], self.model_params["image_size"][1])).to(self.device)
        self.mask = torch.ones(1).long().to(self.device) # ignore the goal
        self.obs_bytes = self.transform_image_to_string(self.image_msg, (160,120))
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.image_msg)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(self.image_msg)

    def get_yaw_delta(self, yaw_reshape):
        yaw_delta = yaw_reshape[-1] - yaw_reshape[0]
        yaw_delta_sign = np.where(yaw_delta >= np.pi, -1, 0)
        yaw_delta_sign = np.where(yaw_delta < -np.pi, 1, yaw_delta_sign)
        yaw_delta = yaw_delta + yaw_delta_sign*2*np.pi
        return yaw_delta

    def process_images(self):
        self.obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=True)
        self.obs_images = torch.split(self.obs_images, 3, dim=1)
        self.obs_images = torch.cat(self.obs_images, dim=1) 
        self.obs_images = self.obs_images.to(self.device)
    
    def infer_actions(self):
        # Get early fusion obs goal for conditioning
        self.obs_cond = self.model('vision_encoder', 
                                    obs_img=self.obs_images, 
                                    goal_img=self.fake_goal, 
                                    input_goal_mask=self.mask)

        # infer action
        with torch.no_grad():
            # encoder vision features
            if len(self.obs_cond.shape) == 2:
                self.obs_cond = self.obs_cond.repeat(self.args.num_samples, 1)
            else:
                self.obs_cond = self.obs_cond.repeat(self.args.num_samples, 1, 1)
            
            # initialize action from Gaussian noise
            self.noisy_action = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
            self.naction = self.noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            self.start_time = time.time()
            for k in self.noise_scheduler.timesteps[:]:
                # predict noise
                self.noise_pred = self.model(
                    'noise_pred_net',
                    sample=self.naction,
                    timestep=k,
                    global_cond=self.obs_cond
                )
                # inverse diffusion step (remove noise)
                self.naction = self.noise_scheduler.step(
                    model_output=self.noise_pred,
                    timestep=k,
                    sample=self.naction
                ).prev_sample
            print("time elapsed:", time.time() - self.start_time)

        self.naction = to_numpy(self.get_action())
        print("sampled actions shape: ", self.naction.shape)
        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_msg.data = np.concatenate((np.array([0]), self.naction.flatten())).tolist()
        self.sampled_actions_pub.publish(self.sampled_actions_msg)
        
    def timer_callback(self):
        self.chosen_waypoint = np.zeros(4, dtype=np.float32)
        if len(self.context_queue) > self.model_params["context_size"]:
            
            self.process_images()
            self.infer_actions()

            if self.step%self.plan_freq == 0:

                result = self.get_traj_from_server(self.obs)

                if result:
                    self.reached_goal = True
                    self.traj_duration = 0
                    print("DONE")
                    exit()

            self.chosen_waypoint = self.chosen_action[self.args.waypoint]

            # Normalize and publish waypoint
            if self.model_params["normalize"]:
                self.chosen_waypoint[:2] *= (self.MAX_V / self.RATE)  

            self.waypoint_msg.data = self.chosen_waypoint.tolist()
            self.execute = 0
            while self.execute < self.args.waypoint:
                self.waypoint_pub.publish(self.waypoint_msg)
                self.execute += 1
                time.sleep(self.timer_period)
            self.blank_msg = Float32MultiArray()
            self.blank_msg.data = np.zeros(4, dtype=np.float32).tolist()
            self.waypoint_pub.publish(self.blank_msg)
            self.step += 1

def main(args):
    rclpy.init()
    low_level_policy = LowLevelPolicy(args)

    rclpy.spin(low_level_policy)
    low_level_policy.destroy_node()
    rclpy.shutdown()
    
    print("Registered with master node. Waiting for image observations...")
    print(f"Using {low_level_policy.device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=5, # close waypoints exhibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=6,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--ip", 
        "-i", 
        default="localhost",
        type=str,
    )
    parser.add_argument(
        "--prompt",
        "-p", 
        default="Go to the kitchen",
        type=str,
        help="Prompt for the policy",
    )
    args = parser.parse_args()
    main(args)


