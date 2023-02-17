import base64
import torch
from PIL import Image
from io import BytesIO
from diffusers import EulerDiscreteScheduler, StableDiffusionInpaintPipeline

def decode_base64(base64_string):
    decoded_string = BytesIO(base64.b64decode(base64_string))
    img = Image.open(decoded_string)
    return img

def model_fn(model_dir):
    # Load stable diffusion and move it to the GPU
    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, 
                                                   scheduler=scheduler,
                                                   revision="fp16",
                                                   torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    #pipe.enable_attention_slicing()
    return pipe


def predict_fn(data, pipe):
    # get prompt & parameters
    prompt = data.pop("inputs", data) 
    # Require json string input. Inference to convert imge to string.
    input_img = data.pop("input_img", data)
    mask_img = data.pop("mask_img", data)
    # set valid HP for stable diffusion
    num_inference_steps = data.pop("num_inference_steps", 25)
    guidance_scale = data.pop("guidance_scale", 6.5)
    num_images_per_prompt = data.pop("num_images_per_prompt", 2)
    image_length = data.pop("image_length", 512)
    # run generation with parameters
    generated_images = pipe(
        prompt,
        image = decode_base64(input_img),
        mask_image = decode_base64(mask_img),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        height=image_length,
        width=image_length,
    #)["images"] # for Stabel Diffusion v1.x
    ).images
    
    # create response
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())
        
    return {"generated_images": encoded_images}


In order to upload the model to an Amazon S3 bucket, it is necessary to first create a model.tar.gz archive. It's crucial to note that the archive should consist of the files directly and not a folder that holds them. For instance, the file should appear as follows:

import tarfile
import os

# helper to create the model.tar.gz
def compress(tar_dir=None,output_file="model.tar.gz"):
    parent_dir=os.getcwd()
    os.chdir(tar_dir)
    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir('.'):
          print(item)
          tar.add(item, arcname=item)    
    os.chdir(parent_dir)
            
compress(str(model_tar))

# After we created the model.tar.gz archive we can upload it to Amazon S3. We will 
# use the sagemaker SDK to upload the model to our sagemaker session bucket.
from sagemaker.s3 import S3Uploader

# upload model.tar.gz to s3
s3_model_uri=S3Uploader.upload(local_path="model.tar.gz", \
        desired_s3_uri=f"s3://{sess.default_bucket()}/finetuned-stable-diffusion-v2-1-inpainting")
