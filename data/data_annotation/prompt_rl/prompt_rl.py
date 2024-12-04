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


# Load in-context examples (training data in this case)
"""
1. Object navigation examples
2. Avoidance examples 
3. Grounding examples 
4. Indoor/outdoor examples
5. Object manipulation examples
6. Contrastive examples"""

examples = []
examples_path = glob.glob("examples/*", recursive=True)
for example in examples_path:
    example_dict = {}
    images = [Image.open(img) for img in glob.glob(example + "/*.png")]
    example_dict["images"] = images
    with open(example + "/labels.txt", "r") as f:
        labels = f.read().split("\n")
    example_dict["labels"] = labels
    example_dict["name"] = example.split("/")[-1]
    examples.append(example_dict)

# define prompts
ORIG_PROMPT = "You are a robot moving through an environment. Provided is the series of egocentric observations you have made. Please describe the trajectory you took taking note of the objects and structures you interact with and your relative motion to them. Provide the descriptions in the form of robot instructions as a list of strings that each describe the entire trajectory."
CRITIQUE_PROMPT_1 = "Here is a series of instructions produced by a language agent describing a robot's trajectory through an environment: "
CRITIQUE_PROMPT_2 = "Here are examples of ground truth or human-generated instructions describing the same robot's trajectory through the environment: "
CRITIQUE_PROMPT_3 = "Please provide feedback on the language agent's instructions and suggest improvements: "
RL_PROMPT = f"Here is the original prompt: '{ORIG_PROMPT}'. Based on the provided feedback, please revise the original prompt to better guide the language agent in generating more accurate instructions: "

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
client = OpenAI(api_key=OPENAI_KEY,organization = ORGANIZATION_ID)
MAX_ITERS = 10
# util functions
def pil_to_base64(img):
    img.save("temp.jpg")
    with open("temp.jpg", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    global ORIG_PROMPT
    iters = 0
    while iters < MAX_ITERS:
        gpt_responses = []
        # For each example, prompt the VLM with the trajectory and the starting prompt
        print("Getting responses from GPT-4o")
        if os.path.exists(f"gpt_responses_{iters}.pkl"):
            with open(f"gpt_responses_{iters}.pkl", "rb") as f:
                gpt_responses = pkl.load(f)
        else:
            for example in examples:
                print("Example: ", example["name"])
                images = example["images"]
                labels = example["labels"]
                images_base64 = [pil_to_base64(img) for img in images]
                image_messages = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}",},} for img in images_base64]
                text_message = {"type": "text", "text": ORIG_PROMPT}
                messages = [text_message] + image_messages
                context = [{"role": "user", "content": messages}]
                response = client.chat.completions.create(model="gpt-4o", messages=context)
                gpt_responses.append(response.choices[0].message.content)

            # Save the responses to a file
            with open(f"gpt_responses_{iters}.pkl", "wb") as f:
                pkl.dump(gpt_responses, f)
            # save as text file for easy viewing
            with open(f"gpt_responses_{iters}.txt", "w") as f:
                for i, response in enumerate(gpt_responses):
                    f.write(f"Example: {examples[i]['name']}\n")
                    f.write(f"Response: {response}\n\n")

        # for each example, have the VLM critique the response and provide feedback based on the ground truth examples
        critiques = []
        history = []
        print("Getting critiques from GPT-4o")
        if os.path.exists(f"critiques_{iters}.pkl"):
            with open(f"critiques_{iters}.pkl", "rb") as f:
                history = pkl.load(f)
        else:
            for i, example in enumerate(examples):
                print("Example: ", example["name"])
                response = gpt_responses[i]
                labels_combined = (' ').join([f'{idx+1}. {label}' for idx, label in enumerate(example["labels"])])
                text_message = {"type": "text", "text": f"{CRITIQUE_PROMPT_1}{response} {CRITIQUE_PROMPT_2} {labels_combined} {CRITIQUE_PROMPT_3}"}
                messages = [text_message]
                history.append({"role": "user", "content": messages})
                context = [{"role": "user", "content": messages}]
                response = client.chat.completions.create(model="gpt-4o", messages=context)
                critiques.append(response.choices[0].message.content)
                history.append({"role": "assistant", "content": [{"type" : "text", "text" : response.choices[0].message.content}]})
            
            # Save the critiques to a file
            with open(f"critiques_{iters}.pkl", "wb") as f:
                pkl.dump(history, f)

        # Ask the VLM to take note of these critiques and modify the original prompt
        # construct history:
        history.append({"role": "user", "content": [{"type": "text", "text": RL_PROMPT, }],})
        response = client.chat.completions.create(model="gpt-4o", messages=history)
        print("Response: ", response.choices[0].message.content)
        ORIG_PROMPT = response.choices[0].message.content
        iters += 1

if __name__ == "__main__":
    main()


