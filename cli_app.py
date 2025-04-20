'''
python cli_app.py chongyun0.png \
    --video_prompt "masterpiece, best quality, highly detailed" \
    --output "chongyun0_sketch.mp4"
'''

import os
import random
import argparse
import uuid
#import cv2
import numpy as np
import torch
from PIL import Image
import wd14tagger
import memory_management
from diffusers_helper.code_cond import unet_add_coded_conds
from diffusers_helper.cat_cond import unet_add_concat_conds
from diffusers_helper.k_diffusion import KDiffusionSampler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers_vdm.pipeline import LatentVideoDiffusionPipeline
from diffusers_vdm.utils import resize_and_center_crop, save_bcthw_as_mp4

# Set up environment
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
result_dir = os.path.join('./', 'results')
os.makedirs(result_dir, exist_ok=True)

class ModifiedUNet(UNet2DConditionModel):
    @classmethod
    def from_config(cls, *args, **kwargs):
        m = super().from_config(*args, **kwargs)
        unet_add_concat_conds(unet=m, new_channels=4)
        unet_add_coded_conds(unet=m, added_number_count=1)
        return m

# Load models
model_name = 'lllyasviel/paints_undo_single_frame'
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(torch.float16)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(torch.bfloat16)
unet = ModifiedUNet.from_pretrained(model_name, subfolder="unet").to(torch.float16)

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

video_pipe = LatentVideoDiffusionPipeline.from_pretrained(
    'lllyasviel/paints_undo_multi_frame',
    fp16=True
)

memory_management.unload_all_models([
    video_pipe.unet, video_pipe.vae, video_pipe.text_encoder, video_pipe.image_projection, video_pipe.image_encoder,
    unet, vae, text_encoder
])

k_sampler = KDiffusionSampler(
    unet=unet,
    timesteps=1000,
    linear_start=0.00085,
    linear_end=0.020,
    linear=True
)

def find_best_bucket(h, w, options):
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in options:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket

@torch.inference_mode()
def encode_cropped_prompt_77tokens(txt: str):
    memory_management.load_models_to_gpu(text_encoder)
    cond_ids = tokenizer(txt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt").input_ids.to(device=text_encoder.device)
    text_cond = text_encoder(cond_ids, attention_mask=None).last_hidden_state
    return text_cond

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def interrogator_process(x):
    return wd14tagger.default_interrogator(x)

def get_image_dimensions(image_path):
    """Get image dimensions without loading full image"""
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)

def determine_target_size(original_width, original_height):
    """Determine target size maintaining aspect ratio with constraints"""
    bucket_options = [(320, 512), (384, 448), (448, 384), (512, 320)]
    best_bucket = find_best_bucket(original_height, original_width, bucket_options)
    return best_bucket  # returns (height, width)

@torch.inference_mode()
def generate_key_frames(input_fg, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg):
    rng = torch.Generator(device=memory_management.gpu).manual_seed(int(seed))

    memory_management.load_models_to_gpu(vae)
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    memory_management.load_models_to_gpu(text_encoder)
    conds = encode_cropped_prompt_77tokens(prompt)
    unconds = encode_cropped_prompt_77tokens(n_prompt)

    memory_management.load_models_to_gpu(unet)
    fs = torch.tensor(input_undo_steps).to(device=unet.device, dtype=torch.long)
    initial_latents = torch.zeros_like(concat_conds)
    concat_conds = concat_conds.to(device=unet.device, dtype=unet.dtype)
    latents = k_sampler(
        initial_latent=initial_latents,
        strength=1.0,
        num_inference_steps=steps,
        guidance_scale=cfg,
        batch_size=len(input_undo_steps),
        generator=rng,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        cross_attention_kwargs={'concat_conds': concat_conds, 'coded_conds': fs},
        same_noise_in_batch=True
    ).to(vae.dtype) / vae.config.scaling_factor

    memory_management.load_models_to_gpu(vae)
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [fg] + pixels + [np.zeros_like(fg) + 255]

    return pixels

@torch.inference_mode()
def process_video_inner(image_1, image_2, prompt, seed=123, steps=25, cfg_scale=7.5, fs=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    frames = 16

    target_height, target_width = find_best_bucket(
        image_1.shape[0], image_1.shape[1],
        options=[(320, 512), (384, 448), (448, 384), (512, 320)]
    )

    image_1 = resize_and_center_crop(image_1, target_width=target_width, target_height=target_height)
    image_2 = resize_and_center_crop(image_2, target_width=target_width, target_height=target_height)
    input_frames = numpy2pytorch([image_1, image_2])
    input_frames = input_frames.unsqueeze(0).movedim(1, 2)

    memory_management.load_models_to_gpu(video_pipe.text_encoder)
    positive_text_cond = video_pipe.encode_cropped_prompt_77tokens(prompt)
    negative_text_cond = video_pipe.encode_cropped_prompt_77tokens("")

    memory_management.load_models_to_gpu([video_pipe.image_projection, video_pipe.image_encoder])
    input_frames = input_frames.to(device=video_pipe.image_encoder.device, dtype=video_pipe.image_encoder.dtype)
    positive_image_cond = video_pipe.encode_clip_vision(input_frames)
    positive_image_cond = video_pipe.image_projection(positive_image_cond)
    negative_image_cond = video_pipe.encode_clip_vision(torch.zeros_like(input_frames))
    negative_image_cond = video_pipe.image_projection(negative_image_cond)

    memory_management.load_models_to_gpu([video_pipe.vae])
    input_frames = input_frames.to(device=video_pipe.vae.device, dtype=video_pipe.vae.dtype)
    input_frame_latents, vae_hidden_states = video_pipe.encode_latents(input_frames, return_hidden_states=True)
    first_frame = input_frame_latents[:, :, 0]
    last_frame = input_frame_latents[:, :, 1]
    concat_cond = torch.stack([first_frame] + [torch.zeros_like(first_frame)] * (frames - 2) + [last_frame], dim=2)

    memory_management.load_models_to_gpu([video_pipe.unet])
    latents = video_pipe(
        batch_size=1,
        steps=int(steps),
        guidance_scale=cfg_scale,
        positive_text_cond=positive_text_cond,
        negative_text_cond=negative_text_cond,
        positive_image_cond=positive_image_cond,
        negative_image_cond=negative_image_cond,
        concat_cond=concat_cond,
        fs=fs
    )

    memory_management.load_models_to_gpu([video_pipe.vae])
    video = video_pipe.decode_latents(latents, vae_hidden_states)
    return video, image_1, image_2

def generate_video(keyframes, prompt, steps, cfg, fps, seed, output_path=None):
    result_frames = []
    cropped_images = []

    for i, (im1, im2) in enumerate(zip(keyframes[:-1], keyframes[1:])):
        frames, im1, im2 = process_video_inner(
            im1, im2, prompt, seed=seed + i, steps=steps, cfg_scale=cfg, fs=3
        )
        result_frames.append(frames[:, :, :-1, :, :])
        cropped_images.append([im1, im2])

    video = torch.cat(result_frames, dim=2)
    video = torch.flip(video, dims=[2])

    if output_path is None:
        uuid_name = str(uuid.uuid4())
        video_filename = os.path.join(result_dir, uuid_name + '.mp4')
        image_filename = os.path.join(result_dir, uuid_name + '.png')
    else:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        video_filename = output_path if output_path.lower().endswith('.mp4') else output_path + '.mp4'
        image_filename = os.path.splitext(video_filename)[0] + '.png'

    Image.fromarray(cropped_images[0][0]).save(image_filename)
    save_bcthw_as_mp4(video, video_filename, fps=fps)
    return video_filename, image_filename

def main():
    parser = argparse.ArgumentParser(description='Generate video from an input image')
    parser.add_argument('input_image', type=str, help='Path to input image file')
    
    # Generation parameters
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--cfg', type=float, default=3.0, help='CFG scale')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    
    # Video generation parameters
    parser.add_argument('--video_steps', type=int, default=50, help='Video generation steps')
    parser.add_argument('--video_cfg', type=float, default=7.5, help='Video CFG scale')
    parser.add_argument('--fps', type=int, default=4, help='Frames per second')
    parser.add_argument('--video_seed', type=int, default=123, help='Video generation seed')
    
    # Prompt parameters
    parser.add_argument('--prompt', type=str, default=None, help='Optional custom prompt (skips auto-prompt generation)')
    parser.add_argument('--video_prompt', type=str, default='1girl, masterpiece, best quality',
                       help='Prompt for video generation (default: "1girl, masterpiece, best quality")')
    
    # Output parameters
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video path (default: random filename in results directory)')
    parser.add_argument('--keep_temp', action='store_true',
                       help='Keep temporary keyframe images')
    
    args = parser.parse_args()

    # Get original image dimensions
    original_width, original_height = get_image_dimensions(args.input_image)
    print(f"Original image dimensions: {original_width}x{original_height}")
    
    # Determine target size maintaining aspect ratio
    target_height, target_width = determine_target_size(original_width, original_height)
    print(f"Using target dimensions: {target_width}x{target_height} (maintaining aspect ratio)")

    # Load input image
    input_fg = np.array(Image.open(args.input_image))

    # Step 1: Generate prompt
    if args.prompt is None:
        print("Generating prompt...")
        prompt = interrogator_process(input_fg)
        print(f"Generated prompt: {prompt}")
    else:
        prompt = args.prompt
        print(f"Using custom prompt: {prompt}")

    # Step 2: Generate key frames
    print("Generating key frames...")
    input_undo_steps = [400, 600, 800, 900, 950, 999]
    n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
    
    key_frames = generate_key_frames(
        input_fg=input_fg,
        prompt=prompt,
        input_undo_steps=input_undo_steps,
        image_width=target_width,
        image_height=target_height,
        seed=args.seed,
        steps=args.steps,
        n_prompt=n_prompt,
        cfg=args.cfg
    )

    # Save key frames temporarily
    temp_key_frames = []
    temp_dir = os.path.join(result_dir, 'temp_' + str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, frame in enumerate(key_frames):
        temp_path = os.path.join(temp_dir, f'keyframe_{i:03d}.png')
        Image.fromarray(frame).save(temp_path)
        temp_key_frames.append(temp_path)

    # Step 3: Generate video
    print("Generating video...")
    print(f"Using video prompt: {args.video_prompt}")
    video_path, preview_path = generate_video(
        #keyframes=[(f,) for f in temp_key_frames],
        keyframes=[np.asarray(Image.open(f)) for f in temp_key_frames],
        prompt=args.video_prompt,
        steps=args.video_steps,
        cfg=args.video_cfg,
        fps=args.fps,
        seed=args.video_seed,
        output_path=args.output
    )

    # Clean up
    if not args.keep_temp:
        print("Cleaning up temporary files...")
        for frame in temp_key_frames:
            os.remove(frame)
        os.rmdir(temp_dir)
    else:
        print(f"Temporary files kept in: {temp_dir}")

    print("\nGeneration complete!")
    print(f"Video saved to: {video_path}")
    print(f"Preview image saved to: {preview_path}")

if __name__ == '__main__':
    main()
