import os 
import glob
import torch 
import clip
from tqdm import tqdm
import pickle as pkl


input_path = "/hdd/sacson_language_rand_15_25"
lang_txt_paths = glob.glob(f"{input_path}/*/traj_data.pkl", recursive=True)
model_version = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

for path in tqdm(lang_txt_paths): 
    print(f"Processing path {path}")

    with open(path, "rb") as old_file:
        traj_data = pkl.load(old_file)

    lang_annotations = traj_data["language_annotations"]
    prompts = clip.tokenize([lang["traj_description"] for lang in lang_annotations]).to(device)
    text_features = model.encode_text(prompts).detach().cpu().numpy()
    new_path = path.replace("traj_data.pkl", "traj_data_w_embed.pkl")
    traj_data[f"text_features"] = text_features
    with open(new_path, "wb") as new_file:
        pkl.dump(traj_data, new_file)
