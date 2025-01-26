from io import BytesIO
from PIL import Image
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys
import inspect
import tensorflow as tf
import wandb

# flask app 
import base64
from flask import Flask, request, jsonify

from datetime import datetime
import os
from collections import deque
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
from typing import Callable, List, Tuple
import imageio
import numpy as np
import google.generativeai
##############################################################################
# Import Gemini parameters
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
client = genai.GenerativeModel(model_name="gemini-1.5-flash")
##############################################################################
app = Flask(__name__)

DEBUG = False 
PRIMITIVES = ["Go forward", "Turn left", "Adjust left", "Turn right", "Adjust right" "Stop"]


def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

@app.route('/suggest_action', methods=["POST"])
def suggest_action():
    # Receive data 
    data = request.get_json()
    img = base64.b64decode(data['image'])
    prompt = data['prompt']

    curr_obs = Image.open(BytesIO(img_data))
    curr_obs.save("curr_obs.png")
    curr_obs_64 = image_to_base64(curr_obs)

    vlm_prompt = f"A robot is moving through an environment and has the task '{prompt}'. Given the current observation, which action in the list {PRIMITIVES} should the robot take next? Return your response as the single action in the list of primitives with no additional information."
    contents = []
    contents.append({"role": "user",
                                    "parts": [
                                        {
                                            "text": vlm_prompt,
                                        },
                                        {
                                            "inline_data" : {
                                                "mime_type" : "image/jpeg",
                                                "data": curr_obs_64,
                                            }
                                        }
                                    ]
                        })

    ai_response = client.generate_content(contents)
    action = ai_response.text
    response = jsonify(action=action)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
