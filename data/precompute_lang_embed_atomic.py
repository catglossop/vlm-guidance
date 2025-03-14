import os 
import glob
import torch 
import clip
from tqdm import tqdm
import pickle as pkl
import tensorflow_hub as hub
import tensorflow_text
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

USE_CLIP = False
USE_UNI = False
USE_T5 = True
REDO = True
# input_path = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/cory_hall_labelled"
# input_path = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/go_stanford_cropped_labelled"
# input_path = "/home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets/sacson_labelled"
input_path = "/home/noam/LLLwL/datasets/atomic_dataset_finer"
lang_txt_paths = glob.glob(f"{input_path}/*/*/*/traj_data.pkl", recursive=True)
if USE_CLIP:
    model_version = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
if USE_UNI:
    text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
if USE_T5:
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5EncoderModel.from_pretrained("google-t5/t5-small")

for path in tqdm(lang_txt_paths): 

    with open(path, "rb") as old_file:
        traj_data = pkl.load(old_file)
    lang_annotations = [traj_data["language_instruction"], traj_data["varied_language_instruction"]]
    if USE_CLIP:
        try:
            prompts = clip.tokenize([lang for lang in lang_annotations]).to(device)
        except: 
            print(f"Error in path {path}")
            print(lang_annotations)
            exit()
        text_features = model.encode_text(prompts).detach().cpu().numpy()
        new_path = path.replace("traj_data.pkl", "traj_data_w_embed_clip.pkl")
    if USE_UNI:
        text_features = text_model([lang for lang in lang_annotations]).numpy()
        new_path = path.replace("traj_data.pkl", "traj_data_w_embed_google.pkl")
    if USE_T5:
        if len(lang_annotations) == 0:
            print(f"Path {path} has no lang annotations")
            continue
        new_path = path.replace("traj_data.pkl", "traj_data_w_embed_t5.pkl")
        if os.path.exists(new_path) and not REDO:
            continue
        text_features = []
        for lang in lang_annotations:
            tokens = tokenizer(lang, return_tensors="pt", padding=True)
            text_feature = model(tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state.mean(dim=1).detach().cpu().numpy()
            text_features.append(text_feature)
        text_features = np.vstack(text_features)
        
    traj_data["text_features"] = text_features
    with open(new_path, "wb") as new_file:
        pkl.dump(traj_data, new_file)

all_paths = glob.glob(f"{input_path}/*", recursive=True)

for path in tqdm(all_paths):
    if not os.path.exists(os.path.join(path, "traj_data.pkl")):
        print(f"Path {path} does not have traj_data.pkl")
        breakpoint()
