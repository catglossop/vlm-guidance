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

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5EncoderModel.from_pretrained("google-t5/t5-small")


lang = "Explore"
tokens = tokenizer(lang, return_tensors="pt", padding=True)
text_feature = model(tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state.mean(dim=1).detach().cpu().numpy()

# Save the text feature 
with open("explore_embed.pkl", "wb") as f:
    pkl.dump(text_feature, f)