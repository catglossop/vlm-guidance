import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable, List
import numpy as np
import yaml
import threading
from PIL import Image as PILImage
import argparse
import time

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from cv_bridge import CvBridge
from utils import to_numpy, transform_images, load_model

# UTILS
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
IMAGE_SIZE = (224, 224)
from data.data_utils import IMAGE_ASPECT_RATIO
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC, 
                        REACHED_GOAL_TOPIC)

class NavigateLocal(Node): 

    def __init__(self, 
                args
                ):
        super().__init__('navigate_vla')
        self.args = args
        self.context_queue = []
        self.context_size = 1
        self.language_prompt = args.prompt

        # Load the config
        self.load_config(ROBOT_CONFIG_PATH)
 
        # Load data config
        self.load_data_config()
        
        # SUBSCRIBERS  
        self.image_msg = Image()
        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            1)
        
        # PUBLISHERS
        self.reached_goal = False
        self.reached_goal_msg = Bool()
        self.reached_goal_pub = self.create_publisher(
            Bool, 
            REACHED_GOAL_TOPIC, 
            1)
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
        self.bridge = CvBridge()
        # TIMERS
        self.timer_period = 1/self.RATE  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    # # Utils
    # def unnormalize_data(self, ndata, stats):
    #     ndata = (ndata + 1) / 2
    #     data = ndata * (stats['max'] - stats['min']) + stats['min']
    #     return data

    # def get_action(self):
    #     # diffusion_output: (B, 2*T+1, 1)
    #     # return: (B, T-1)
    #     ndeltas = self.nactions
    #     ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    #     ndeltas = to_numpy(ndeltas)
    #     ndeltas = self.unnormalize_data(ndeltas, self.ACTION_STATS)
    #     actions = np.cumsum(ndeltas, axis=1)
    #     return torch.from_numpy(actions).to(self.device)
    
    def transform_images_vla(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False):
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
    
    def transform_images_viz(self, pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
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

    def compare_output(self):
        dataset_name = "sacson"
        traj_1 = self.nactions
        prompt_1 = self.language_prompt
        viz_img = self.transform_images_viz(self.context_queue[-1], IMAGE_SIZE) 
        fig, ax = plt.subplots(1, 2)
        if len(traj_1.shape) > 2:
            trajs = [*traj_1]
        else:
            trajs = [traj_1]
        start_pos = np.array([0,0])
        goal_pos = np.array([0,0])
        plot_trajs_and_points(
            ax[0], 
            trajs,
            [start_pos, goal_pos], 
            traj_colors=[CYAN] + [MAGENTA]*(self.num_samples - 1),
            point_colors=[GREEN, RED],
            traj_labels = ["gt"] + ["pred"]*(self.num_samples -1)
        )
        plot_trajs_and_points_on_image(      
            ax[1],
            np.transpose(viz_img.numpy().squeeze(0), (1,2,0)),
            dataset_name,
            trajs,
            [start_pos, goal_pos],
            traj_colors=[CYAN] + [MAGENTA]*(self.num_samples - 1),
            point_colors=[GREEN, RED],
        )
        ax[0].legend([prompt_1])
        ax[1].legend([prompt_1])
        ax[0].set_ylim((-5, 5))
        ax[0].set_xlim((-5, 15))
        plt.savefig("visualize.png")

    def load_config(self, robot_config_path):
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.MAX_V = robot_config["max_v"]
        self.MAX_W = robot_config["max_w"]
        self.VEL_TOPIC = "/task_vel"
        self.DT = 1/robot_config["frame_rate"]
        self.RATE = robot_config["frame_rate"]
        self.EPS = 1e-8
        self.WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
        self.FLIP_ANG_VEL = np.pi/4
    
    def load_data_config(self):
        # LOAD DATA CONFIG
        with open(os.path.join(os.path.dirname(__file__), DATA_CONFIG), "r") as f:
            data_config = yaml.safe_load(f)
        # POPULATE ACTION STATS
        self.ACTION_STATS = {}
        for key in data_config['action_stats']:
            self.ACTION_STATS[key] = np.array(data_config['action_stats'][key])

    def image_callback(self, msg):
        self.image_msg = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.image_msg = PILImage.fromarray(self.image_msg)
        img = self.image_msg.save("test_image.jpg")
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.image_msg)
            else:
                os.makedirs("context_queue", exist_ok=True)
                for i in range(len(self.context_queue)):
                    self.context_queue[i].save(f"context_queue/{i}.jpg") 
                self.context_queue.pop(0)
                self.context_queue.append(self.image_msg)
    
    def process_images(self):
        self.obs_images = transform_images_vla(self.context_queue, self.model_params["image_size"], center_crop=False)
    
    def infer_actions(self):
        print("Getting VLA output")
        obs_base64 = image_to_base64(Image.fromarray(self.context_queue))
        req_str = server_address + str("/gen_action")
        response = requests.post(req_str, json={'obs': obs_base64, 'prompt':self.language_prompt}, timeout=99999999)
        ndeltas = np.array(response.json()['action'])
        self.nactions = ndeltas.reshape(-1, 2)
        # ndeltas = self.unnormalize_data(ndeltas, self.ACTION_STATS)
        # self.nactions = np.cumsum(ndeltas, axis=1)
        self.naction = self.nactions

        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_msg.data = np.concatenate((np.array([0]), self.nactions.flatten())).tolist()
        self.sampled_actions_pub.publish(self.sampled_actions_msg)

        self.chosen_waypoint = self.naction[self.args.waypoint, :] 
        
    def timer_callback(self):
        start = time.time()
        self.chosen_waypoint = np.zeros(4, dtype=np.float32)
        if len(self.context_queue) >= self.context_size:

            # Process observations
            start_image_time = time.time()
            self.process_images()
            print("Process time: ", time.time() - start_image_time)
            
            # Use policy to get actions
            start_infer_time = time.time()
            self.infer_actions()
            print("Infer time: ", time.time() - start_infer_time)

            # Visualize actions 
            start_viz_time = time.time()
            self.compare_output()
            print("Compare time: ", time.time() - start_viz_time)

        # Normalize and publish waypoint
        if self.model_params["normalize"]:
            self.chosen_waypoint[:2] *= (self.MAX_V / self.RATE)  
        print("Chosen waypoint shape: ", self.chosen_waypoint.shape)
        print("Chosen waypoint: ", self.chosen_waypoint)
        self.waypoint_msg.data = self.chosen_waypoint.tolist()
        self.execute = 0
        # while self.execute < self.args.waypoint:
        self.waypoint_pub.publish(self.waypoint_msg)
        # self.execute += 1
        # time.sleep(self.timer_period)
        # self.blank_msg = Float32MultiArray()
        # self.blank_msg.data = np.zeros(4, dtype=np.float32).tolist()
        # self.waypoint_pub.publish(self.blank_msg)
        print("Elapsed time: ", time.time() - start )

def main(args):
    rclpy.init()
    nav_policy = NavigateLocal(args)

    rclpy.spin(nav_policy)
    nav_policy.destroy_node()
    rclpy.shutdown()
    
    print("Registered with master node. Waiting for image observations...")
    print(f"Using {nav_policy.device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run local language conditioned navigation")
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument("--num-samples",
        "-n",
        default=1,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--model-type",
        "-m", 
        default="rft", 
        type=str,
        help="Model type to use",
    )
    parser.add_argument(
        "--prompt",
        "-p", 
        default="Go to the kitchen",
        type=str,
        help="Prompt for the policy",
    )
    parser.add_argument(
        "--server-address",
        "-s",
        default="http://localhost:5000",
        type=str,
        help="Server address for inference",
    )
    args = parser.parse_args()
    main(args)


