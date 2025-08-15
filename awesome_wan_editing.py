# 1) define
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import imageio
import requests
from PIL import Image
import argparse
import omegaconf

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline # pipeline_wan.py
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from utils.wan_attention import register_attention_processor

# 1) load video
# helper function to load videos
def load_video(file_path: str):
    images = []
    
    # Assuming it's a local file path
    vid = imageio.get_reader(file_path)

    fps = vid.get_meta_data()['fps']
    #print("fps:", fps)

    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)

    return images, fps

def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a Training-Free Editing script for WAN2.1.")

    parser.add_argument('--config', type=str, default='./config/object_editing/bear_edit.yaml')
    args = parser.parse_args()

    config = omegaconf.OmegaConf.load(args.config)
    return config

if __name__ == '__main__':

    config = get_args()

    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    ## Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    ## Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P (1.3B only)
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16) # dtype checking
    pipe.scheduler = scheduler
    pipe.to("cuda")

    # load video
    video, fps = load_video(config['video']['video_path']) # horsejump-high

    print(len(video)) # 49
    print(fps) # 16.0
    print(video[0].size) # (832, 480)

    ########################################### prompt
    source_prompt = config['video']['source_prompt']
    target_prompt = config['video']['target_prompt']

    source_blend = config['video']['source_blend'] # foreground object
    target_blend = config['video']['target_blend'] # foreground object

    ## source prompt
    source_blend_idx = pipe.tokenizer(source_blend,
                padding="max_length",
                max_length=512, # wan2.1 setting
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
                ).input_ids
    #source_blend_embed = pipe.text_encoder(source_blend_idx.cuda()).last_hidden_state # (1, 512, 4096)
    #print(source_blend_idx)
    print(source_blend_idx.shape) # (1, 512)
    source_blend_idx = source_blend_idx[0]
    #print(pipe.tokenizer.decode(source_blend_idx[0]))
    #print(torch.nonzero(source_blend_idx, as_tuple=True))
    source_blend_idx = source_blend_idx[torch.nonzero(source_blend_idx, as_tuple=True)][:-1]
    print(pipe.tokenizer.decode(source_blend_idx), '//', source_blend_idx)

    source_prompt_idx = pipe.tokenizer(source_prompt,
                padding="max_length",
                max_length=512, # wan2.1 setting
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
                ).input_ids[0]
    #print(source_prompt_idx)
    idx_list = torch.where(source_prompt_idx==source_blend_idx[0])[0]
    for idx in idx_list:
        boolean = torch.all(source_prompt_idx[idx:idx+len(source_blend_idx)]==source_blend_idx)

        if boolean:
            source_idx = list(range(idx, idx+len(source_blend_idx)))
            break
    print(source_idx)

    ## target prompt
    target_blend_idx = pipe.tokenizer(target_blend,
                padding="max_length",
                max_length=512, # wan2.1 setting
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
                ).input_ids
    #print(target_blend_idx)
    print(target_blend_idx.shape) # (1, 512)
    target_blend_idx = target_blend_idx[0]
    target_blend_idx = target_blend_idx[torch.nonzero(target_blend_idx, as_tuple=True)][:-1]
    print(pipe.tokenizer.decode(target_blend_idx), '//', target_blend_idx)

    target_prompt_idx = pipe.tokenizer(target_prompt,
                padding="max_length",
                max_length=512, # wan2.1 setting
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
                ).input_ids[0]
    #print(target_prompt_idx)
    idx_list = torch.where(target_prompt_idx==target_blend_idx[0])[0]
    for idx in idx_list:
        boolean = torch.all(target_prompt_idx[idx:idx+len(target_blend_idx)]==target_blend_idx)

        if boolean:
            target_idx = list(range(idx, idx+len(target_blend_idx)))
            break
    print(target_idx)
    ###########################################

    # inference
    if config['training-free-type']['flag_flowalign']:

        # model register
        ## hyperparameters
        num_inference_steps = config['infernece']['num_inference_step']
        strength = config['flowalign']['strength']
        total_steps = int(num_inference_steps*strength)
        target_guidance_scale = config['flowalign']['target_guidance_scale']
        fg_zeta_scale = config['flowalign']['zeta_scale'] # foreground (attn-masking) or all (no attn-masking)
        bg_zeta_scale = config['flowalign']['bg_zeta_scale'] # background
        num_frames = len(video)
        print("Total steps:", total_steps)

        start_step = 1
        end_step = total_steps
        start_layer = 1
        end_layer = 30
        layer_idx = None
        total_layers = 30

        masking_flag = config['flowalign']['flag_attnmask']
        masking_layer = [11, 12, 13, 14, 15, 16, 17]

        ## replace attn processor
        # AttributeError: 'WanTransformer3DModel' object has no attribute 'set_attn_processor'
        # Be careful, kyujinpy modified WanTransformer3DModel
        # Add `set_attn_processor` and `attn_processor` functions in above class.
        register_attention_processor(pipe.transformer, 
                                    processor_type="MasaCtrlProcessor",
                                    start_step=start_step,
                                    end_step=end_step,
                                    start_layer=start_layer, # attention layer
                                    end_layer=end_layer,
                                    layer_idx=layer_idx,
                                    total_layers=total_layers, # WAN2.1-1.3B has 30 layers / 14B: ??
                                    total_steps=total_steps,
                                    masking=masking_flag, # apply cross-attn masking
                                    masking_layer=masking_layer,
                                    frame_num=num_frames,
                                    source_idx=source_idx,
                                    target_idx=target_idx,
                                    )
                                    
        output = pipe.flowalign(
            video = video,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            height=480,
            width=832,
            num_inference_steps=num_inference_steps,
            strength=strength,
            target_guidance_scale=target_guidance_scale, # [5.0, 7.5, 10.0, 13.5]
            fg_zeta_scale=fg_zeta_scale,
            bg_zeta_scale=bg_zeta_scale,
        ).frames[0]

        export_to_video(output, config['flowalign']['save_video'], fps=16)

        print(output.shape)
        print("WANAlign2.1 Finish!!")
    
    elif config['training-free-type']['flag_flowedit']:

        register_attention_processor(pipe.transformer, 
                                    processor_type="WanAttnProcessor2_0",
                                    )
        
        # model register
        ## hyperparameters
        num_inference_steps = config['infernece']['num_inference_step']
        strength = config['flowedit']['strength']
        total_steps = int(num_inference_steps*strength)
        target_guidance_scale = config['flowedit']['target_guidance_scale']
        source_guidance_scale = config['flowedit']['source_guidance_scale']
        num_frames = len(video)
        print("Total steps:", total_steps)  

        output = pipe.flowedit(
            video = video,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            height=480,
            width=832,
            num_inference_steps=num_inference_steps,
            strength=strength,
            target_guidance_scale=target_guidance_scale,
            source_guidance_scale=source_guidance_scale,
        ).frames[0]

        export_to_video(output, config['flowedit']['save_video'], fps=16)

        print(output.shape)
        print("WANEdit2.1 Finish!!")
    
    else:
        raise Exception("Not support.")