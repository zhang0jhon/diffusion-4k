import torch
import os

import argparse

from pipeline_flux_4k_wavelet import FluxPipeline

img_prompts = [
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.", \
            "An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in thought pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down shirt , he wears a brown beret and glasses and has a very professorial appearance.", \
]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoint/flux_wavelet",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./visualization/flux_wavelet",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0, 
        help="the FLUX.1 dev variant is a guidance distilled model, default 3.5 for original FLUX.1 dev",
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
        default=50,
        help=(
            "inference steps."
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help=(
            "max sequence length for t5."
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
    
    pipe = FluxPipeline.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16)

    pipe = pipe.to("cuda")
    # pipe.enable_model_cpu_offload()

    if args.prompt is not None:
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
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
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                partitioned=args.partitioned,
            ).images[0]

            model_name = os.path.basename(args.checkpoint_path)

            image.save(f"{args.save_path}/{model_name}_gs_{args.guidance_scale}_{args.height}x{args.width}_seed_{args.seed}_prompt_{idx}.jpg", quality=95)


if __name__ == "__main__":
    args = parse_args()
    main(args)