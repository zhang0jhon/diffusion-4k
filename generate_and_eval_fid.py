import torch
import os

import json
import argparse

import clip
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024**2)
Image.MAX_IMAGE_PIXELS = None

from pipeline_flux_4k_wavelet import FluxPipeline # for F16 VAEs
from pipeline_sd3_4k_wavelet import StableDiffusion3Pipeline # for F16 VAEs
# for diffusers import FluxPipeline, StableDiffusion3Pipeline # for F8 VAEs

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score
from functools import partial

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def fid_preprocess(image, size):
    from torchvision.transforms import functional as F
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) #/ 255.0
    image = F.resize(image, size)
    return F.center_crop(image, (size, size))

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
        default="./evaluation/flux_wavelet",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0, # 5.0 
        help="the FLUX.1 dev variant is a guidance distilled model, default 3.5 for original FLUX.1 dev",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help=(
            "The size for generated image."
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
        "--json_file",
        type=str,
        default="./Aesthetic-4K/eval/size_2048/metadata.jsonl",  # "./Aesthetic-4K/eval/size_4096/metadata.jsonl"
        help="Json file for validation.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    os.makedirs(args.save_path, exist_ok=True) 
    
    if "flux" in args.checkpoint_path:
        pipe = FluxPipeline.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16) 
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16)

    pipe = pipe.to("cuda")
    # pipe.enable_model_cpu_offload()

    # CLIPScore Model
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    def calculate_clip_score(images, prompts):
        clipscore = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
        return float(clipscore)
    
    # FID Model
    fid = FrechetInceptionDistance(normalize=False)

    # Aesthetics Model
    clip_model, clip_preprocess = clip.load("ViT-L/14", device="cuda") # clip model to compute aesthetic score
    aesthetic_model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    aesthetic_model.load_state_dict(torch.load("./improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"))
    aesthetic_model.to("cuda")
    aesthetic_model.eval()

    clip_scores, aesthetic_scores = [], []

    with open(f"{args.json_file}", "r") as f:
        json_data = f.readlines()
        for d in json_data:
            data = json.loads(d)

            file_name = data["file_name"]
            prompt = data["text"]

            real_image = Image.open(f"{os.path.dirname(args.json_file)}/{file_name}").convert('RGB')
            width, height = real_image.size

            # flux
            image = pipe(
                prompt,
                height=args.size,
                width=args.size,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length if "flux" in args.checkpoint_path else 77,
                generator=torch.Generator("cpu").manual_seed(args.seed)
            ).images[0]

            # compute FID
            fid.update(fid_preprocess(np.array(real_image), size=args.size), real=True) # resize and center crop to (size, size)
            fid.update(fid_preprocess(np.array(image), size=args.size), real=False) # resize and center crop to (size, size)

            # compute CLIPScore
            clipscore = calculate_clip_score(np.expand_dims(np.array(image), 0), [prompt])

            # compute Aesthetics
            aesthetic_image = clip_preprocess(image).unsqueeze(0).to("cuda")
            with torch.no_grad():
                image_features = clip_model.encode_image(aesthetic_image)
            im_emb_arr = normalized(image_features.cpu().detach().numpy())
            aesthetic_score = aesthetic_model(torch.from_numpy(im_emb_arr).to("cuda").type(torch.cuda.FloatTensor)).detach().cpu().numpy()[0][0]

            clip_scores.append(clipscore)
            aesthetic_scores.append(aesthetic_score)

            # image.save(f"{args.save_path}/{os.path.basename(file_name)}", quality=95)
            image.save(f"{args.save_path}/{os.path.splitext(os.path.basename(file_name))[0]}.jpg", quality=95)

            print(prompt)
            print("clip score: {}, aesthetic score: {}".format(clipscore, aesthetic_score))

        # compute FID
        fid_score = fid.compute().detach().cpu().numpy()

        print("FID: {}, CLIPScore: {}, Aesthetics: {}".format(fid_score, np.mean(clip_scores), np.mean(aesthetic_scores)))

if __name__ == "__main__":
    args = parse_args()
    main(args)