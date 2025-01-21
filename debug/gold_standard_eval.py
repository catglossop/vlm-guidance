from PIL import Image
import os
import argparse
import clip
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import List, Tuple, Dict, Optional
import glob 
import tensorflow_hub as hub
import tensorflow_text
from transformers import T5EncoderModel, T5Tokenizer
import torch
import pickle as pkl
from torchvision import transforms
import torchvision.transforms.functional as TF
import model
import train
from model.model import ResNetFiLMTransformer
from model.lelan.lnp_comp import LNP_comp, LNP_clip_FiLM, LNPMultiModal
from model.lelan.lnp import LNP_clip, LNP, DenseNetwork_lnp, DenseNetwork, LNP_MM
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os
path = os.path.abspath(train.__file__)
print(path)
from train.training.train_utils import replace_bn_with_gn, model_output_diffusion_eval
from train.training.train_eval_loop import load_model
from train.training.train_utils import model_output
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
IMAGE_SIZE = (96, 96)

# load data_config.yaml
with open(os.path.join("/home/noam/LLLwL/lcbc/data/data_config.yaml"), "r") as f:
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

def t5_embed(text, device):
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5EncoderModel.from_pretrained("google-t5/t5-small")
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    text_features = model(tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state.mean(dim=1)
    return text_features

def compare_output(traj_1, traj_2, gt_trajs, viz_imgs, prompt_1, prompt_2, model_name):
    dataset_name = "sacson"
    fig, ax = plt.subplots(2, 2, figsize=(20,20))
    start_pos = np.array([0,0])
    goal_pos_1 = gt_trajs[0][-1]
    goal_pos_2 = gt_trajs[1][-1]
    for i in range(traj_1.shape[0]):
        curr_trajs = [traj_1[i], gt_trajs[0]]
        plot_trajs_and_points(
            ax[0,0], 
            curr_trajs,
            [start_pos, goal_pos_1], 
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
        plot_trajs_and_points_on_image(      
            ax[1,0],
            np.transpose(viz_imgs[0].numpy(), (1,2,0)),
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
            [start_pos, goal_pos_2], 
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
        plot_trajs_and_points_on_image(      
            ax[1,1],
            np.transpose(viz_imgs[1].numpy(), (1,2,0)),
            dataset_name,
            curr_trajs,
            [start_pos, goal_pos_2],
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
    ax[0,0].set_ylim((-5, 10))
    ax[0,0].set_xlim((-5, 10))
    ax[0,1].set_ylim((-5, 10))
    ax[0,1].set_xlim((-5, 10))
    ax[0,0].set_title(prompt_1)
    ax[0,1].set_title(prompt_2)
    ax[0,0].get_legend().remove()
    ax[0,1].get_legend().remove()
    prompt_1_joined = ("_").join(prompt_1.split())
    prompt_2_joined = ("_").join(prompt_2.split())
    output_path = f"outputs/{model_name}/{prompt_1_joined}_vs_{prompt_2_joined}.png"
    plt.savefig(output_path)
    plt.show()

    return output_path

def load_config(config_path):
    with open("/home/noam/LLLwL/lcbc/config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    return config


def model_output_lnp(model, noise_scheduler, context, goal_img, prompt_embedding, prompt, pred_horizon, action_dim, num_samples, batch_size, device, mask_image=False):
    try:
        categorical = model.action_head.embedding_dim == 4
    except:
        categorical = False
    output = model_output_diffusion_eval(
        model,
        noise_scheduler,
        context.clone(),
        prompt_embedding.float(),
        [prompt],
        goal_img,
        pred_horizon,
        action_dim,
        num_samples,
        batch_size,
        device,
        mask_image,
        categorical,
    )
    return output["actions"].detach().cpu().numpy()
        
def main(args): 
    config = load_config(args.config)
    print(config["context_size"])
    # Load the context for each eval path 

    if args.eval_path_1:
        prompt_1 = args.eval_path_1.split("/")[-1]
        traj_data_1 = np.load(os.path.join(args.eval_path_1, "traj_data.pkl"), allow_pickle=True)
        pos_1 = np.array(traj_data_1["position"])
        pos_1 = pos_1 - pos_1[0]
        pos_1[:, [1, 0]] = pos_1[:, [0, 1]]
    if args.eval_path_2:
        prompt_2 = args.eval_path_2.split("/")[-1]
        traj_data_2 = np.load(os.path.join(args.eval_path_2, "traj_data.pkl"), allow_pickle=True)
        pos_2 = np.array(traj_data_2["position"])
        pos_2 = pos_2 - pos_2[0]
        pos_2[:, [1, 0]] = pos_2[:, [0, 1]]
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
    elif config["model_type"] == "lnp":
        if config["vision_encoder"] == "lnp_clip_film":
            vision_encoder = LNP_clip_FiLM(
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                lang_encoding_size=config["lang_encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                )
            vision_encoder = replace_bn_with_gn(vision_encoder)

        noise_scheduler = DDPMScheduler(
                num_train_timesteps=config["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork_lnp(embedding_dim=config["encoding_size"]*(config["context_size"]+1), control_horizon=config["len_traj_pred"])
        model = LNP_clip(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )    
    elif config["model_type"] == "lnp_multi_modal":
        if config["vision_encoder"] == "lnp_multi_modal":
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
    checkpoint_path = os.path.join(args.model_path, f"{args.checkpoint}.pth")

    latest_checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(1)) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
    load_model(model, config["model_type"], latest_checkpoint)
    model.to(args.device)
    model.eval()
    if config["language_encoder"] == "clip":
        prompt_embedding_1 = clip_embed(prompt_1, args.device).to(torch.float).to(args.device)
        prompt_embedding_2 = clip_embed(prompt_2, args.device).to(torch.float).to(args.device)
    elif config["language_encoder"] == "t5":
        prompt_embedding_1 = t5_embed(prompt_1, args.device).to(torch.float).to(args.device)
        prompt_embedding_2 = t5_embed(prompt_2, args.device).to(torch.float).to(args.device)

    num_imgs = min(len(glob.glob(os.path.join(args.eval_path_1, "*.jpg"))), len(glob.glob(os.path.join(args.eval_path_2, "*.jpg"))))
    viz_images = []
    for idx in range(num_imgs - config["context_size"]):
        print(f"On step: {idx} of {num_imgs - config['context_size']}")
        if args.eval_path_1:
            context_1 = [Image.open(os.path.join(args.eval_path_1, str(i)+".jpg")) for i in range(idx, idx + config["context_size"] + 1)]
        if args.eval_path_2:
            context_2 = [Image.open(os.path.join(args.eval_path_2, str(i)+".jpg")) for i in range(idx, idx + config["context_size"] + 1)]

        context_1_transf = transform_images(context_1, IMAGE_SIZE).to(args.device)
        context_2_transf = transform_images(context_2, IMAGE_SIZE).to(args.device)
        viz_1 = transform_images(context_1[-1], VISUALIZATION_IMAGE_SIZE)[0] 
        viz_2 = transform_images(context_2[-1], VISUALIZATION_IMAGE_SIZE)[0]
        with torch.no_grad():
            if config["model_type"] == "rft":
                output_1 = model(context_1_transf.clone(), prompt_embedding_1).detach().cpu().numpy()
                output_2 = model(context_2_transf.clone(), prompt_embedding_2).detach().cpu().numpy()
            elif config["model_type"] == "lnp":
                output_1 = model_output_lnp(model, noise_scheduler, context_1_transf.clone(), None, prompt_embedding_1, config["len_traj_pred"], 2, args.num_samples, 1, args.linear_output, args.device)
                output_2 = model_output_lnp(model, noise_scheduler, context_2_transf.clone(), None, prompt_embedding_2, config["len_traj_pred"], 2, args.num_samples, 1, args.linear_output, args.device)
            elif config["model_type"] == "lnp_multi_modal":
                mask_image = True
                goal_img = torch.zeros((1, 3, 96, 96)).to(args.device)
                output_1 = model_output_lnp(model, noise_scheduler, context_1_transf.clone(), goal_img, prompt_embedding_1, prompt_1, config["len_traj_pred"], 2, args.num_samples, 1, args.device, mask_image)
                output_2 = model_output_lnp(model, noise_scheduler, context_2_transf.clone(), goal_img, prompt_embedding_2, prompt_2, config["len_traj_pred"], 2, args.num_samples, 1, args.device, mask_image)
            model_name = args.model_path.split("/")[-1]
            os.makedirs(f"outputs/{model_name}", exist_ok=True)
            viz_path = compare_output(output_1, output_2, [pos_1, pos_2], [viz_1, viz_2], prompt_1, prompt_2, model_name)
            curr_img = Image.open(viz_path)
            viz_images.append(curr_img)
        
        viz_images[0].save(f"outputs/{model_name}/{prompt_1}_vs_{prompt_2}.gif", save_all=True, append_images=viz_images[1:], duration=1000, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to model", default="/home/noam/LLLwL/lcbc/logs/lcbc/lelan_multi_modal_2024_12_06_15_44_19")
    parser.add_argument("--eval_path_1", type=str, help="path to eval data", default="gnm_version/avoid_the_person")
    parser.add_argument("--eval_path_2", type=str, help="path to eval data", default="gnm_version/move_toward_the_person")
    parser.add_argument("--config", type=str, help="path to config file", default="/home/noam/LLLwL/lcbc/config/transformer_interleaved_atomic_full.yaml")
    parser.add_argument("--checkpoint", type=int, help="checkpoint to load", default="15")
    parser.add_argument("--num_samples", type=int, help="number of samples", default=8)
    args = parser.parse_args()
    device = "cuda:1" if torch.cuda.is_available() else None
    args.device = device

    main(args)

