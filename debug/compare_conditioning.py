from PIL import Image
import os
import argparse
import clip
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import List, Tuple, Dict, Optional
import glob 

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from model.model import ResNetFiLMTransformer
from train.training.train_eval_loop import load_model
from train.training.train_utils import model_output
from data.data_utils import transform_images, IMAGE_ASPECT_RATIO, VISUALIZATION_IMAGE_SIZE
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
IMAGE_SIZE = (96, 96)

# load data_config.yaml
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)

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

def clip_embed(text, device): 
    model_version = "ViT-B/32"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(text).to(device)
    text_features = model.encode_text(text)
    return text_features

def compare_output(traj_1, traj_2, viz_img, prompt_1, prompt_2):
    dataset_name = "sacson"
    fig, ax = plt.subplots(1, 2)
    if len(traj_1.shape) > 2:
        trajs = [*traj_1, *traj_2]
    else:
        trajs = [traj_1, traj_2]
    start_pos = np.array([0,0])
    goal_pos = np.array([0,0])
    plot_trajs_and_points(
        ax[0], 
        trajs,
        [start_pos, goal_pos], 
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    plot_trajs_and_points_on_image(      
        ax[1],
        np.transpose(viz_img.numpy(), (1,2,0)),
        dataset_name,
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    ax[0].legend([prompt_1, prompt_2])
    ax[1].legend([prompt_1, prompt_2])
    ax[0].set_ylim((-5, 20))
    ax[0].set_xlim((-5, 20))
    plt.savefig("comparison.png")
    plt.show()

def load_config(config_path):
    with open("../config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    return config

def main(args): 
    config = load_config(args.config)
    if args.random_image:
        searching = True 
        possible_imgs = glob.glob(f"{args.image_path}/*/*.jpg", recursive=True)
        while searching: 
            current_path = np.random.choice(possible_imgs)
            print(current_path)
            current_img = Image.open(current_path)
            response = input("Use this image? (y/n): ")
            plt.imshow(current_img)
            plt.show()
            idx = int(current_path.split("/")[-1].strip(".jpg"))
            if response == "y" and idx >= config["context_size"]: 
                args.prompt_1 = input("Enter prompt 1: ")
                args.prompt_2 = input("Enter prompt 2: ")
                searching = False
            else: 
                continue
        args.image_path = ("/").join(current_path.split("/")[:-1])
        print(args.image_path)
        args.start_idx = int(current_path.split("/")[-1].strip(".jpg")) - config["context_size"]
    else:
        args.start_idx = args.image_path.split("/")[-1].strip(".jpg")

    
    if config["model_type"] == "rft":
        model = ResNetFiLMTransformer(
            config["efficientnet_model"],
            config["context_size"],
            config["len_traj_pred"],
            config["encoding_size"],
            config["lang_encoding_size"],
            config["mha_num_attention_layers"],
            config["mha_num_attention_heads"],
            config["vocab_size"],
            config["dropout"],
            args.device,
        )
    latest_path = os.path.join(args.model_path, "latest.pth")
    latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
    load_model(model, latest_checkpoint)
    model = model.to(args.device)
    model.eval()

    prompt_embedding_1 = clip_embed(args.prompt_1, args.device).to(torch.float).to(args.device)
    prompt_embedding_2 = clip_embed(args.prompt_2, args.device).to(torch.float).to(args.device)
    context_orig = []
    for i in range(args.start_idx, args.start_idx+config["context_size"]+1):
        try:
            context_orig.append(Image.open(os.path.join(args.image_path, str(i)+".jpg")))
        except:
            context_orig.append(Image.open(os.path.join(args.image_path, str(i)+".png")))
    context = transform_images(context_orig, IMAGE_SIZE).to(args.device)  
    viz_img = transform_images(context_orig[-1], VISUALIZATION_IMAGE_SIZE)[0]  
    with torch.no_grad():
        output_2 = model(context, prompt_embedding_2).detach().cpu().numpy()
        output_1 = model(context, prompt_embedding_1).detach().cpu().numpy()

    compare_output(output_1, output_2, viz_img, args.prompt_1, args.prompt_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to model")
    parser.add_argument("--prompt_1", type=str, help="prompt to generate")
    parser.add_argument("--prompt_2", type=str, help="prompt to generate")
    parser.add_argument("--image_path", type=str, help="path to images")
    parser.add_argument("--start_idx", type=int, help="start index of context")
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--random_image", action="store_true")
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    args.device = device

    main(args)

