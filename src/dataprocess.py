# Author : Sujay
# Description  : External Data Processor (need to run externally)
# Date created : 5th Dec 2023

import torch
import PIL
import pandas as pd
import os
from PIL import Image


DATA_NORMAL_PATH = "../data/train/NORMAL"
OUTPUT_NORMAL_PATH = "../data/train/resized/NORMAL_resized"

DATA_PNEU_PATH = "../data/train/PNEUMONIA"
OUTPUT_PNEU_PATH = "../data/train/resized1/PNEUMONIA_resized"
TARGET_SIZE = (64, 64)


os.makedirs(OUTPUT_NORMAL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PNEU_PATH, exist_ok=True)

normal_images = os.listdir(DATA_NORMAL_PATH)
pneu_images = os.listdir(DATA_PNEU_PATH)

SAMPLE_NORMAL_SIZE = len(normal_images)
SAMPLE_PNEU_SIZE = len(pneu_images)

if ".DS_Store" in normal_images:
    normal_images.remove(".DS_Store")

if ".DS_Store" in pneu_images: 
    pneu_images.remove(".DS_Store")

n = 0
for i in normal_images:
    if n==SAMPLE_NORMAL_SIZE:
        break
    input_image_path = os.path.join(DATA_NORMAL_PATH, i)
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(TARGET_SIZE)
    output_image_path = os.path.join(OUTPUT_NORMAL_PATH, i)
    resized_image.save(output_image_path)
    print(f"Original size: {original_image.size}, Resized size: {resized_image.size}")
    n = n+1

n = 0
for i in pneu_images:
    if n==SAMPLE_PNEU_SIZE:
        break
    input_image_path = os.path.join(DATA_PNEU_PATH, i)
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(TARGET_SIZE)
    output_image_path = os.path.join(OUTPUT_PNEU_PATH, i)
    resized_image.save(output_image_path)
    print(f"Original size: {original_image.size}, Resized size: {resized_image.size}")
    n = n+1




