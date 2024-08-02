from base64 import b64encode

import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login

# For video display:
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os
from utils import (tokenizer, text_encoder, torch_device, scheduler, generate_with_embs,
                   set_timesteps, latents_to_pil, unet, get_output_embeds, plot_image)

torch.manual_seed(1)

if not (Path.home()/'.cache/huggingface'/'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()


# prompt = ["A colorful dancer, nat geo photo"]
# height = 512                        # default height of Stable Diffusion
# width = 512                         # default width of Stable Diffusion
# num_inference_steps = 50            # Number of denoising steps
# guidance_scale = 8                  # Scale for classifier-free guidance
# generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
# batch_size = 1
#
#
# # Prep text (same as before)
# text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
# with torch.no_grad():
#     text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
# max_length = text_input.input_ids.shape[-1]
# uncond_input = tokenizer(
#     [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
# )
# with torch.no_grad():
#     uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
# text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
#
#
# # Prep Scheduler (setting the number of inference steps)
# set_timesteps(scheduler, num_inference_steps)
#
#
# # Prep latents (noising appropriately for start_step)
# start_step = 10
# start_sigma = scheduler.sigmas[start_step]
# noise = torch.randn_like(encoded)
# latents = scheduler.add_noise(encoded, noise, timesteps=torch.tensor([scheduler.timesteps[start_step]]))
# latents = latents.to(torch_device).float()



# Loop
# for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
#     if i >= start_step: # << This is the only modification to the loop we do
#
#         # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
#         latent_model_input = torch.cat([latents] * 2)
#         sigma = scheduler.sigmas[i]
#         latent_model_input = scheduler.scale_model_input(latent_model_input, t)
#
#         # predict the noise residual
#         with torch.no_grad():
#             noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
#
#         # perform guidance
#         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#
#         # compute the previous noisy sample x_t -> x_t-1
#         latents = scheduler.step(noise_pred, t, latents).prev_sample
#
# latents_to_pil(latents)[0]


style_embed = torch.load('learned_embeds.bin')

prompt = 'A picture of a puppy'
# Tokenize
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
input_ids = text_input.input_ids.to(torch_device)


# Access the embedding layer
token_emb_layer = text_encoder.text_model.embeddings.token_embedding

# Get token embeddings
token_embeddings = token_emb_layer(input_ids)

# The new embedding - special style
replacement_token_embedding = style_embed[''].to(torch_device)

# The new embedding. In this case just the input embedding of token 2368...mixing CAT
# replacement_token_embedding = text_encoder.get_input_embeddings()(torch.tensor(2368, device=torch_device))

# Insert this into the token embeddings (
token_embeddings[0, torch.where(input_ids[0]==6829)] = replacement_token_embedding.to(torch_device)

# get pos embed
pos_emb_layer = text_encoder.text_model.embeddings.position_embedding
position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
position_embeddings = pos_emb_layer(position_ids)

# Combine with pos embs
input_embeddings = token_embeddings + position_embeddings

#  Feed through to get final output embs
modified_output_embeddings = get_output_embeds(input_embeddings)

image = generate_with_embs(modified_output_embeddings, text_input)

plot_image(image)