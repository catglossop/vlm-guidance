import requests
from io import BytesIO
from PIL import Image
import numpy as np
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys

import inspect

import numpy as np
import tensorflow as tf
from absl import app, flags

import wandb

# jax diffusion stuff
from absl import app as absl_app
from absl import flags
from PIL import Image

# flask app here
import base64
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

from datetime import datetime
import os
from collections import deque
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
from typing import Callable, List, Tuple
import imageio
import numpy as np
from absl import app, flags
from openai import OpenAI
##############################################################################
# Import OpenAI params 
gpt_model = "gpt-4o"
##############################################################################
app = Flask(__name__)


OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
client = OpenAI(api_key=OPENAI_KEY,
                    organization = ORGANIZATION_ID)
gpt_model = gpt_model
message_buffer = []
DEBUG = False 
PRIMITIVES = ["Go forward", "Turn left", "Turn right", "Stop"]
BUFFER_SIZE = 10

def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

plan_message_buffer = []
@app.route('/gen_plan', methods=["POST"])
def gen_ll_plan():
    global plan_message_buffer
    # Receive data 
    data = request.get_json()
    img_data = base64.b64decode(data['actions'])
    hl_prompt = data['hl_prompt']

    curr_obs = Image.open(BytesIO(img_data))
    curr_obs.save("curr_obs.png")
    curr_obs_64 = image_to_base64(curr_obs)

    planning_context = f"""A small roomba-like robot is moving through an indoor environment. It can only move through the environment and cannot manipulate objects. The robot is currently executing the task '{hl_prompt}'. 
                           Given the current observation and this task, decompose the high level task into subtasks that can be completed to achieve this task. 
                           For example, if the high level task is "Go to the kitchen.", the task might be decomposed into ["turn down the hallway", "go down the hallway", "GEN_NEW_PLAN"] 
                           where the "GEN_NEW_PLAN" token indicates where the plan is not yet certain. There should be less than 5 subtasks in this plan. If it seems that the high level task has been completed (ie. the object has been reached and is approximately less than 0.5 meters away or the task is done), 
                           set 'task_success' to True in your response. Give your response as a json object with keys 'plan', 'task_success', and 'reason'. The 'plan' field should be a list of subtasks as strings, the 'task_success' field should be a boolean, and 'reason' should be a string description of the reason for the plan."""
    planning_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": planning_context},
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(curr_obs_64)},
        },
        ],
        }
    plan_message_buffer.append(planning_message)
    ai_response = client.chat.completions.create(
            model=gpt_model,
            messages=plan_message_buffer,
            max_tokens=300,
    )
    plan_message_buffer.append(ai_response.choices[0].message)
    if len(message_buffer) > BUFFER_SIZE: 
        plan_message_buffer = plan_message_buffer[-BUFFER_SIZE:]

    plan = ai_response.choices[0].message.content
    response = jsonify(plan=plan)
    return response

action_message_buffer = []
@app.route('/verify_action', methods=["POST"])
def verify_action():
    global action_message_buffer
    # Receive data 
    data = request.get_json()
    img_data = base64.b64decode(data['actions'])
    ll_prompt = data['ll_prompt']
    hl_prompt = data['hl_prompt']

    curr_obs = Image.open(BytesIO(img_data))
    curr_obs.save("curr_obs.png")
    curr_obs_64 = image_to_base64(curr_obs)
    action_context = f"""A robot is moving through an indoor environment. The robot has been tasked with the high level task '{hl_prompt}' and is executing the subtask {ll_prompt} to complete this task. 
                           We provide an annotated version of the robot's current observation with trajectories it can take projected onto the image in cyan, magenta, yellow, green, blue, and red. 
                           Select the trajectory which will lead the robot to complete the subtask. If none of the trajectories immediately accomplish the task '{hl_prompt}', select the trajectory which will help the robot
                           explore the environment to complete the current subtask. If it seems that the task has been completed (ie. the object has been reached and is approximately less than 0.5 meters away or the task is done), 
                           set 'task_success' to True in your response. Give your response as a json object with keys 'trajectory', 'task_success', and 'reason'. The 'trajectory' field should be a single word for the color of the selected trajectory, the 'task_success' field should be a boolean, and 'reason' should be a string description of the reason for the selection."""
    action_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": action_context},
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(curr_obs_64)},
        },
        ],
        }
    action_message_buffer.append(action_message)
    ai_response = client.chat.completions.create(
            model=gpt_model,
            messages=action_message_buffer,
            max_tokens=300,
    )
    action_message_buffer.append(ai_response.choices[0].message)
    if len(action_message_buffer) > BUFFER_SIZE: 
        action_message_buffer = action_message_buffer[-BUFFER_SIZE:]

    selected_trajectory = ai_response.choices[0].message.content
    print(selected_trajectory)
    response = jsonify(traj=selected_trajectory)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
