import cv2
import os, sys
import numpy as np
from PIL import Image

from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--img_root",
        type=str,
        default="./evaluation/flux_wavelet",
        help="Path to generated image folder.",
    )
    parser.add_argument(
        "--prop",
        type=str,
        default="entropy",
        choices=["entropy", "contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
        help="prop for GLCM.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help=(
            "The size for generated images."
        ),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help=(
            "The patch size for GLCM Score."
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def gray_proc(img):
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    table64 = np.array([(i//4) for i in range(256)]).astype("uint8")
    gray64 = cv2.LUT(gimg, table64)
    return gray64

def glcm(gray64, prop='entropy'):
    dist = [1, 2, 3, 4]
    degree = [0, np.pi/4, np.pi/2, np.pi*3/4]
    glcm = graycomatrix(gray64, dist, degree, levels=64, normed=True)

    if prop=='entropy':
        score = shannon_entropy(glcm)
        return score
    feature = graycoprops(glcm, prop).round(4)
    score = np.mean(feature)
    return score


# import subprocess

# def get_folder_size(path):
#     result = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
#     return int(result.stdout.split()[0])

def main(args):

    img_scores, img_sizes = [], []
    img_list = os.listdir(args.img_root)
    origin_size = args.size * args.size * 3 / 1024 / 1024 # h*w*c  (MB)
    
    # print(f"Folder size: {get_folder_size(args.img_root)/1024/1024} MB")

    for img_name in img_list:
        img = cv2.imread(os.path.join(args.img_root, img_name))
        img_size = os.path.getsize(os.path.join(args.img_root, img_name)) / 1024 / 1024 # MB

        gray64 = gray_proc(img)

        scores = 0
        count  = 0
        
        for i in range(0, int(args.size/args.patch_size)):
            for j in range(0, int(args.size/args.patch_size)):
                testi = gray64[i*args.patch_size:i*args.patch_size+args.patch_size, j*args.patch_size:j*args.patch_size+args.patch_size]
                scores = scores + glcm(testi, prop=args.prop)
                count = count + 1

        scores = scores / count
        print(img_name, scores, img_size)

        img_scores.append(scores)

        img_sizes.append(img_size)

    print("GCLM Score: {}".format(np.mean(img_scores)))

    print("Compression Ratio: {}".format(origin_size * len(img_sizes) / np.sum(img_sizes)))

if __name__ == "__main__":
    args = parse_args()
    main(args)


# python eval.py --img_root /data/yizhou/VAND2.0/zjj/diffusion-4k/visualization/flux_jpg/size_2048/generate_images/q95/


# python eval.py --img_root /data/yizhou/VAND2.0/zjj/diffusion-4k/visualization/sd3_jpg/size_2048/generate_images/q95/


# python eval.py --img_root /data/yizhou/VAND2.0/zjj/diffusion-4k/visualization/sd3_wavelet_jpg/size_2048/generate_images/q95/



# python eval.py --img_root ./evaluation/flux_wavelet_v2