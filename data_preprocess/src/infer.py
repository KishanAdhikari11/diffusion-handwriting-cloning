import argparse
import torch
from path import Path
import os
import matplotlib.pyplot as plt
from dataloader import DataLoaderImgFile
from eval import evaluate
from net import WordDetectorNet
from visualization import visualize_and_plot
import numpy as np
from PIL import Image
import cv2

def preprocess_to_black_and_white(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    
    # Optional: Apply slight median blur to reduce noise
    binary = cv2.medianBlur(binary, 3)
    
    return binary

def draw_bounding_boxes(img, aabbs, output_dir, img_index):
    # Create a subdirectory for each image
    image_output_dir = os.path.join(output_dir, f"image_{img_index}")
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Ensure image array is in uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    img_height, img_width = img.shape[:2]
    img_pil = Image.fromarray(img)
    
    for idx, aabb in enumerate(aabbs, start=1):
        x1 = max(0, min(img_width, int(aabb.xmin)))
        y1 = max(0, min(img_height, int(aabb.ymin)))
        x2 = max(0, min(img_width, int(aabb.xmax)))
        y2 = max(0, min(img_height, int(aabb.ymax)))
        
        print(f"Bounding box {idx}: ({x1}, {y1}, {x2}, {y2})")
        
        if x1 < x2 and y1 < y2:
            # Crop the region
            cropped_word = img_pil.crop((x1, y1, x2, y2))
            
            # Convert PIL Image to numpy array for OpenCV processing
            cropped_array = np.array(cropped_word)
            
            # Convert to black and white
            bw_cropped = preprocess_to_black_and_white(cropped_array)
            
            # Convert back to PIL Image
            bw_pil = Image.fromarray(bw_cropped)
            
            # Create white background
            white_bg = Image.new("L", bw_pil.size, color=255)
            white_bg.paste(bw_pil, (0, 0))
            
            # Save as JPG
            output_path = os.path.join(image_output_dir, f"{idx}.jpg")
            white_bg.save(output_path, "JPEG", quality=95)
            print(f"Saved cropped word {idx} at: {output_path}")
        else:
            print(f"Invalid bounding box {idx} -- Skipped")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    
    # Load the model
    net = WordDetectorNet()
    net.load_state_dict(torch.load('../model/weights', map_location=args.device))
    net.eval()
    net.to(args.device)
    
    # Prepare data loader
    loader = DataLoaderImgFile(Path('../data/test/denoised_images'), net.input_size, args.device)
    res = evaluate(net, loader, max_aabbs=1000)
    
    output_dir = "../english"
    
    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        f = loader.get_scale_factor(i)
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]
        
        img = loader.get_original_img(i)
        
        draw_bounding_boxes(img, aabbs, output_dir, i)

if __name__ == '__main__':
    main()