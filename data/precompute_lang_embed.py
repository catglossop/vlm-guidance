import os 
import glob
import torch 
import clip
from tqdm import tqdm
import pickle as pkl
import tensorflow_hub as hub
import tensorflow_text

USE_CLIP = False
USE_UNI = True
# input_path = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/cory_hall_labelled"
# input_path = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/go_stanford_cropped_labelled"
input_path = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/sacson_labelled"
lang_txt_paths = glob.glob(f"{input_path}/*/traj_data.pkl", recursive=True)
if USE_CLIP:
    model_version = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
if USE_UNI:
    text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

for path in tqdm(lang_txt_paths): 

    with open(path, "rb") as old_file:
        traj_data = pkl.load(old_file)
    if "text_features" in traj_data.keys():
        continue
    lang_annotations = traj_data["language_annotations"]
    if USE_CLIP:
        try:
            prompts = clip.tokenize([lang["traj_description"] for lang in lang_annotations]).to(device)
        except: 
            print(f"Error in path {path}")
            print(lang_annotations)
            exit()
        text_features = model.encode_text(prompts).detach().cpu().numpy()
        new_path = path.replace("traj_data.pkl", "traj_data_w_embed_clip.pkl")
    if USE_UNI:
        text_features = text_model([lang["traj_description"] for lang in lang_annotations]).numpy()
        new_path = path.replace("traj_data.pkl", "traj_data_w_embed_google.pkl")
    traj_data["text_features"] = text_features
    with open(new_path, "wb") as new_file:
        pkl.dump(traj_data, new_file)

all_paths = glob.glob(f"{input_path}/*", recursive=True)

for path in tqdm(all_paths):
    if not os.path.exists(os.path.join(path, "traj_data.pkl")):
        print(f"Path {path} does not have traj_data.pkl")
        breakpoint()
