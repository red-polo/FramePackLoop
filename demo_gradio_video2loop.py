from diffusers_helper.hf_login import login

import os
import random

hf_home = os.environ.get('HF_HOME')
if hf_home is None:
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import re

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

import torchvision
import torchvision.io as io

def save_bcthw_as_png(x, output_filename):
    # UIと合わせる
    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), 0, 1) * 255
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, 'b c t h w -> c (b h) (t w)')
    torchvision.io.write_png(x, output_filename)
    return output_filename

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

def loop_worker( head_video, prompt, n_prompt, generation_count, seed,connection_second_length, loop_num, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, reduce_file_output, without_preview, output_latent_image):
    for generation_count_index in range(generation_count):
        if stream.input_queue.top() == 'end':
            stream.output_queue.push(('end', None))
            break
        if generation_count != 1:
            seed = random.randint(0, 2**32 - 1)
        stream.output_queue.push(('generation count', f"Generation index:{generation_count_index + 1}"))
        worker(head_video, prompt, n_prompt, generation_count, seed,connection_second_length, loop_num, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, reduce_file_output, without_preview, output_latent_image)
    stream.output_queue.push(('end', None))
    
@torch.no_grad()
def worker(input_video, prompt, n_prompt, generation_count, seed,connection_second_length, loop_num, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, reduce_file_output, without_preview, output_latent_image):
    if stream.input_queue.top() == 'end':
        stream.output_queue.push(('end', None))
        return
    
    if reduce_file_output:
        tmp_filename = "system_preview.mp4"

    
    connection_latent_sections = connection_second_length

    # print(total_latent_sections)
    # print(connection_latent_sections)
    
    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    latent_input_file = None
    try:
        if latent_input_file is None:
            # Clean GPU
            if not high_vram:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

            # Text encoding

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

            if not high_vram:
                fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
                load_model_as_complete(text_encoder_2, target_device=gpu)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Processing input image

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

            # H, W, C = input_image.shape
            # height, width = find_nearest_bucket(H, W, resolution=640)
            # input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            # if not without_preview:
            #     Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_{seed}.png'))

            # input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            # input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
            

            # VAE encoding

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            #start_latent = vae_encode(input_image_pt, vae)



            # VAE encoding
            # 動画と音声を読み込む
            input_video_frames, audio_frames, info = io.read_video(input_video, pts_unit='sec')
            # input_video_frames = input_video_frames.float() / 127.5 - 1
            # input_video_frames = input_video_frames.permute(3, 0, 1, 2).unsqueeze(0)
            # save_bcthw_as_png(input_video_frames, "input.png")

            # latent = vae_encode(input_video_frames, vae)
            # history_pixels = vae_decode(latent, vae).cpu()
            # save_bcthw_as_png(history_pixels, "output.png")

            
            #iput_video_latent_num = int(round(input_video_frames.shape[0] / 4))
            input_video_latent_num = int((input_video_frames.shape[0] + 3) / 4)
            input_video_frames = input_video_frames[:((input_video_latent_num * 4) - 3)]
            
            if input_video_latent_num < 3:
                stream.output_queue.push(('end', None))
                return
            else:
                iput_video_N = min(16, (input_video_latent_num - 3))

            interp_input_frame_length = (iput_video_N + 3)  * 4 - 3
            remain_input_farme_length = max(0, input_video_frames.shape[0] - interp_input_frame_length)

            #入力動画の後ろ
            head_remain_video, head_video = input_video_frames.split([remain_input_farme_length,interp_input_frame_length], dim=0)
            #入力動画の前
            tail_video, tail_remain_video = input_video_frames.split([interp_input_frame_length,remain_input_farme_length], dim=0)
            #入力動画の前
            remain_input_farme_length = max(0, input_video_frames.shape[0] - interp_input_frame_length * 2)
            #tail_video, remain_video, head_video = input_video_frames.split([interp_input_frame_length,remain_input_farme_length,interp_input_frame_length], dim=0)
            
            head_N = iput_video_N
            tail_N = iput_video_N
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            head_video_pt = head_video.float() / 127.5 - 1
            head_video_pt = head_video_pt.permute(3, 0, 1, 2).unsqueeze(0)

            height = head_video_pt.shape[3]
            width = head_video_pt.shape[4]
            
            tail_video_pt = tail_video.float() / 127.5 - 1
            tail_video_pt = tail_video_pt.permute(3, 0, 1, 2).unsqueeze(0)

            
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            head_video_latent = vae_encode(head_video_pt, vae)
            tail_video_latent = vae_encode(tail_video_pt, vae)
            #print(head_video_latent.shape)
            #print(tail_video_latent.shape)



            input_video_frames = input_video_frames[1:]
            tail_video_latent = tail_video_latent[:,:,1:]
            tail_N -= 1
            
            
            # CLIP Vision

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode( input_video_frames[-1,:,:,:].numpy(), feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Dtype

            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

            # Sampling

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3

            ##コネクション作成


            #post_history_latents = torch.zeros(size=(1, 16, 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            head_history_latents = head_video_latent.cpu()
            post_history_latents = tail_video_latent.cpu()
            post_history_pixels = None
            #print(post_history_latents.shape)
            post_total_generated_latent_frames = 0
            print("generate post history")
            
            latent_paddings = [i for i in reversed(range(connection_latent_sections))]
            #print(latent_paddings)
            if connection_latent_sections > 4:
                # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
                # items looks better than expanding it when total_latent_sections > 4
                # One can try to remove below trick and just
                # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
                latent_paddings = [3] + [2] * (connection_latent_sections - 3) + [1, 0]
            
            for latent_padding in latent_paddings:
                print("latent_padding")
                
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    return

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

                indices = torch.arange(0, sum([1,latent_padding_size, latent_window_size, 1, 2, tail_N])).unsqueeze(0)
                clean_latent_indices_pre,blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                    indices.split([1,latent_padding_size, latent_window_size, 1, 2, tail_N], dim=1)
                
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                clean_latent_2x_indices = torch.cat([clean_latent_2x_indices], dim=1)
                clean_latent_4x_indices = torch.cat([clean_latent_4x_indices], dim=1)


                clean_latents_pre  = head_history_latents[:, :, -1:, :, :]
                clean_latents_post, clean_latents_2x, clean_latents_4x = post_history_latents[:, :, :1 + 2 + tail_N, :, :].split([1, 2, tail_N], dim=2)

                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                clean_latents_2x = torch.cat([clean_latents_2x], dim=2)
                clean_latents_4x = torch.cat([clean_latents_4x], dim=2)

            
                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('User ends the task.')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = f'Now making connection video.'
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                    return

                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    # shift=3.0,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                if is_last_section:
                    generated_latents = torch.cat([head_history_latents.to(generated_latents), generated_latents], dim=2)

                post_total_generated_latent_frames += int(generated_latents.shape[2])
                post_history_latents = torch.cat([generated_latents.to(post_history_latents), post_history_latents], dim=2)
                
                post_real_history_latents = post_history_latents[:, :, :post_total_generated_latent_frames, :, :]


                if not without_preview:
                    if not high_vram:
                        offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                        load_model_as_complete(vae, target_device=gpu)

    
                    # if post_history_pixels is None:
                    #     post_history_pixels = vae_decode(post_real_history_latents, vae).cpu()
                    # else:
                    #     section_latent_frames = (latent_window_size * 2) if is_last_section else (latent_window_size * 2)
                    #     overlapped_frames = latent_window_size * 4 - 3

                    #     current_pixels = vae_decode(post_real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    #     #print(current_pixels.shape)
                    #     #print(post_history_pixels.shape)
                    #     post_history_pixels = soft_append_bcthw(current_pixels, post_history_pixels, overlapped_frames)
                        
                    if not high_vram:
                        unload_complete_models()

                    # if reduce_file_output:
                    #     output_filename = os.path.join(outputs_folder, tmp_filename)
                    # else:    
                    #     output_filename = os.path.join(outputs_folder, f'{job_id}_{post_total_generated_latent_frames}_{seed}_post.mp4')

                    # save_bcthw_as_mp4(post_history_pixels, output_filename, fps=30, crf=mp4_crf)

                    # print(f'Decoded. Current latent shape {post_real_history_latents.shape}; pixel shape {post_history_pixels.shape}')

                    # stream.output_queue.push(('file', output_filename))

                if is_last_section:
                    break

            total_generated_latent_frames = post_total_generated_latent_frames

            #1ループ作成
            
            #print(all_latent_section)

            #print(history_latents.shape)
            connection_hisotry_latents = post_real_history_latents[:,:,:latent_window_size*connection_latent_sections,:,:]
            main_history_latents = post_history_latents
            

            # print(main_history_latents.shape)
            # print(connection_hisotry_latents.shape)
            
            final_latents = main_history_latents
            
            if output_latent_image:
                to_pixcel_latents = torch.cat([main_history_latents,
                                connection_hisotry_latents],dim=2)
                
                to_pixcel_latents_png = vae_decode_fake(to_pixcel_latents)
                output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}_{seed}_latent.png')
                save_bcthw_as_png(to_pixcel_latents_png, output_filename)

                output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}_{seed}_latent.pt')
                torch.save(final_latents,output_filename)
                return
        
        else:
            final_latents = torch.load(latent_input_file)   
            filename = os.path.basename(latent_input_file)    
            match = re.match(r"(\d+_\d+_\d+_\d+)_([0-9]+)_([0-9]+)_latent\.pt", filename)

            job_id = match.group(1)  # '250502_204151_409_9042'
            total_generated_latent_frames = match.group(2)  # '9'
            seed = match.group(3)  # '31337'
        
        all_latent_section = connection_latent_sections

        if not high_vram:
            offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
            load_model_as_complete(vae, target_device=gpu)
        #final_latents = final_latents.repeat(1,1,loop_num,1,1)
        #print(final_latents.shape)
        final_history_pixels = None
        MAX = all_latent_section + 2
        pixel_map = dict()
        #print(MAX)
        is_last_section = False
        for i in reversed(range(MAX)):
            #print(i)
            if i == 0:
                is_last_section = True
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            percentage = int(100.0 * i / MAX)
            hint = f'Make Connect {i}/{MAX}'
            desc = f'Now making connected video decoding'
            stream.output_queue.push(('progress', (None, desc, make_progress_bar_html(percentage, hint))))

            
            if i == 0:
                latent_offset = 0
                latent_window = head_N + 3
            elif i == MAX-1:
                latent_offset = head_N + 3 + (i-1)*latent_window_size
                latent_window = tail_N + 3
            else:
                latent_offset = head_N + 3 + (i-1)*latent_window_size
                latent_window = latent_window_size
            
            section_latent_frames = (latent_window_size + latent_window)
            
            if final_history_pixels is None:
                #if pixel_map.get(latent_index) is None:
                decode_latents = final_latents[:,:,latent_offset:]
                final_history_pixels = vae_decode(decode_latents, vae).cpu()
                #print(decode_latents.shape)
                #print(final_history_pixels.shape)
                current_pixels = final_history_pixels
                #pixel_map[latent_index] = final_history_pixels
            else:   
                decode_latents = final_latents[:, :, latent_offset:latent_offset + section_latent_frames]
                current_pixels = vae_decode(decode_latents, vae).cpu()
                #print(decode_latents.shape)
                #print(current_pixels.shape)

                overlapped_frames = latent_window_size * 4 - 3
                if overlapped_frames > final_history_pixels.shape[2]:
                    overlapped_frames = final_history_pixels.shape[2]

                final_history_pixels = soft_append_bcthw(current_pixels, final_history_pixels, overlapped_frames)
            #print(final_history_pixels.shape)

        # ループ１素材だけを取るために前１セクションフレームと、後ろ１セクションフレーム - 3を削除
        final_history_pixels = final_history_pixels[:,:,((head_N + 3) * 4 ) - 3:,:,:]
        final_history_pixels = final_history_pixels[:,:,:-((tail_N + 3) * 4 ),:,:]

    

        input_video_pt = input_video_frames.float() / 127.5 - 1
        input_video_pt = input_video_pt.permute(3, 0, 1, 2).unsqueeze(0)
        #print(head_remain_video_pt.shape)
        final_history_pixels = torch.cat([input_video_pt, final_history_pixels],dim =2 )

        print(final_history_pixels.shape)
        output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}_{seed}_1loop_{loop_num}.mp4')
        save_bcthw_as_mp4(final_history_pixels, output_filename, fps=30, crf=mp4_crf)
        final_history_pixels = final_history_pixels.repeat(1,1,loop_num,1,1)
        output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}_{seed}_loop_{loop_num}.mp4')

        save_bcthw_as_mp4(final_history_pixels, output_filename, fps=30, crf=mp4_crf)
        
        stream.output_queue.push(('file', output_filename))
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
    return


def process(head_video, prompt, n_prompt, generation_count, seed, connection_second_length, loop_num, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, progress_preview_option):
    global stream
    # 表示名と値のマッピング
    options = {
        "All Progress File Output": 1,
        "Reduce Progress File Output": 2,
        "Without Preview": 3,
        "Without VAE Decode": 4,
        "Decode Latent File": 5
    }

    if options[progress_preview_option] == 1:
        reduce_file_output = False
        without_preview = False
        output_latent_image = False
        latent_input_file = None
    elif options[progress_preview_option] == 2:
        reduce_file_output = True
        without_preview = False
        output_latent_image = False
        latent_input_file = None
    elif options[progress_preview_option] == 3:
        reduce_file_output = True
        without_preview = True
        output_latent_image = False
        latent_input_file = None
    elif options[progress_preview_option] == 4:
        reduce_file_output = True
        without_preview = True
        output_latent_image = True
        latent_input_file = None
    elif options[progress_preview_option] == 5:
        reduce_file_output = True
        without_preview = True
        output_latent_image = False

    if options[progress_preview_option] != 5:
        assert head_video is not None, 'No input image!'
    else:
        assert latent_input_file is not None, 'No input Lantet file!'




    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), ''

    stream = AsyncStream()

    async_run(loop_worker, head_video, prompt, n_prompt, generation_count, seed,connection_second_length, loop_num, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, reduce_file_output, without_preview, output_latent_image)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()
        
        if flag == 'generation count':
            generation_count_index = data
            yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), generation_count_index

        if flag == 'end':
            yield output_filename, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), ''
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePackVideo2LoopVideo')
    
    with gr.Row():
        with gr.Column():
            head_video = gr.Video(label="Input Video", autoplay=True, show_share_button=False, height=320, loop=True)
    
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)
           
            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)
                

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                
                generation_count = gr.Slider(label="Generation Count", minimum=1, maximum=500, value=1, step=1,
                                                  info='生成回数です。この値が2以上の場合、Seedはランダムな値が使用されます。')
        
                progress_preview_option = gr.Radio(
                    choices=["All Progress File Output",
                            "Reduce Progress File Output",
                            "Without Preview",
                            ],        
                    label="Progress Option",
                    info = "経過動画のプレビューとファイルの保存方式を設定します。説明は下記。",
                    value="Reduce Progress File Output"                        # ←ここでデフォルト選択！
                )
                gr.Markdown("""          
                    - **All Progress File Output**: **（現在経過ファイルは出力されません）** すべての経過ファイルを出力します。  
                    - **Reduce Progress File Output**:**（現在経過ファイルの出力とプレビューは表示されません）** 途中経過のファイルを同じ名前で上書き保存し、出力ファイル数を減らします。  
                            outputフォルダに system_preview.mp4 というファイルが生成され、プレビュー用に使用されます。  
                            ※動画生成中はこのファイルを開かないでください。  
                    - **Without Preview**: 途中経過のプレビューは出力されません。入力画像や経過ファイルも保存されず、最終的な出力のみが行われます。そのため、最終アウトプットの生成速度がやや向上します。
                    """,)
                #reduce_file_output = gr.Checkbox(label='Reduce File Output', value=False, info='途中経過のファイルの出力を減らします.ただし、outputフォルダにsystem_progress_preview.mp4というファイルが生成されます。これは、プレビューに使用する動画ファイルです。動画生成中は開かないでください。')
                #without_preview = gr.Checkbox(label='Without Preview', value=False, info='途中経過の動画のプレビューが出力されません。その代わりに、最終アウトプットの出力が早くなります。')

                seed = gr.Number(label="Seed", value=31337, precision=0, info='この値はGeneration Countが1の時のみ有効です。')

                #total_second_length = gr.Slider(label="Main Video Length (Section)", minimum=1, maximum=120, value=1, step=1)
                connection_second_length = gr.Slider(label="Connection Video Length (Section)", minimum=1, maximum=120, value=1, step=1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change

                loop_num = gr.Slider(label="Loop Num", minimum=1, maximum=100, value=5, step=1, info='Loop num.')

                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            progress_gcounter = gr.Markdown('', elem_classes='no-generating-animation')

    ips = [head_video, prompt, n_prompt, generation_count, seed, connection_second_length, loop_num, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, progress_preview_option]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, progress_gcounter])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)