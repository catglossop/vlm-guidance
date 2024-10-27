# Load model directly
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import glob
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
path = "/home/noam/LLLwL/datasets/gnm_dataset/sacson/Dec-06-2022-bww8_00000015_4"

describe_prompt = "You are a wheeled robot driving through an environment with a camera mounted on you to see the environment. The image provoided is an observation from the camera. Describe 1) The objects you see in the image, including their color, and their relative position to you, 2) The structures you see, such as walls, rooms, windows and their relative position to you, 3) Any people or semantically significant aspects of the environment. Be specific about whether something is close, far, left, or right of you."
summarize_prompt_p1 = "You are a wheeled robot driving through an environment with a camera mounted on you to see the environment. This list describes a series of observations you made in the environment: "
summarize_prompt_p2 = "From these descriptions, generate 3-4 instructions that the robot could have been following to make these observations. The instructions should be over the entire trajectory. Output the instructions in a json format. The json should have a key 'instructions' with a list of strings as the value. Each string should be an instruction. It should also have a key 'reasoning' that describes why the instructions are reasonable."

# Summarize the descriptions into instructions
def molmo(image, model, processor, prompt):

    # process the image and text
    inputs = processor.process(
        images=[image],
        text=prompt,
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print the generated text
    return generated_text

def main():
    image_paths = glob.glob(f"{path}/*.jpg")[:20]
    image_paths = image_paths[::2]
    print("Number of images: ", len(image_paths))
    # Describe all images
    descs = []
    for image_path in image_paths:
        image = Image.open(image_path)
        desc = molmo(image, model, processor, describe_prompt)
        print("========================================")
        print(desc)
        descs.append(desc)

    summarize_prompt = summarize_prompt_p1
    for i, desc in enumerate(descs):
        summarize_prompt += f" Observation {i}: " + desc + " "
    summarize_prompt += summarize_prompt_p2

    final_desc = molmo(image, model, processor, summarize_prompt)
    print("----------------------------------------")
    print(final_desc)


if __name__ == "__main__":
    main()