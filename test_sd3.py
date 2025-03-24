import torch
import os

import argparse

from pipeline_sd3_4k_wavelet import StableDiffusion3Pipeline

img_prompts = [
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.", \
]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoint/sd3_wavelet",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./visualization/sd3_wavelet",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0, 
        help="guidance scale for stable diffusion",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=4096,
        help=(
            "The height for generated image."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4096, 
        help=(
            "The width for generated image."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help=(
            "inference steps."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--partitioned",
        default=True,
        help=(
            "Partitioned VAE for last convolution layer."
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for image generation.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args



def main(args):
    os.makedirs(args.save_path, exist_ok=True) 

    pipe = StableDiffusion3Pipeline.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16)

    pipe = pipe.to("cuda")
    # pipe.enable_model_cpu_offload()

    if args.prompt is not None:
        image = pipe(
            args.prompt,
            negative_prompt="",
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            partitioned=args.partitioned,
        ).images[0]

        truncated_length = 64
        if len(args.prompt) > truncated_length:
            truncated_prompt = args.prompt[:truncated_length]
        else:
            truncated_prompt = args.prompt

        image.save(f"{args.save_path}/{truncated_prompt}_gs_{args.guidance_scale}_{args.height}x{args.width}_seed_{args.seed}.jpg", quality=95)

    else:
        for idx, img_prompt in enumerate(img_prompts):
            image = pipe(
                img_prompt,
                negative_prompt="",
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                partitioned=args.partitioned,
            ).images[0]

            model_name = os.path.basename(args.checkpoint_path)

            image.save(f"{args.save_path}/{model_name}_gs_{args.guidance_scale}_{args.height}x{args.width}_seed_{args.seed}_prompt_{idx}.jpg", quality=95)


if __name__ == "__main__":
    args = parse_args()
    main(args)