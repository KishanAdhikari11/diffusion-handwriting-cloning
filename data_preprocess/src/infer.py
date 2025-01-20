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

def draw_bounding_boxes(img, aabbs, output_dir, img_index):
    # Create a subdirectory for each image
    image_output_dir = os.path.join(output_dir, f"image_{img_index}")
    os.makedirs(image_output_dir, exist_ok=True)

    # Ensure image array is in uint8 format and 3 channels (RGB)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    if len(img.shape) == 2:  # Grayscale image
        img = np.stack((img,) * 3, axis=-1)  # Convert to RGB

    img_height, img_width = img.shape[:2]
    img_pil = Image.fromarray(img)

    for idx, aabb in enumerate(aabbs, start=1):  
        x1 = max(0, min(img_width, int(aabb.xmin)))
        y1 = max(0, min(img_height, int(aabb.ymin)))
        x2 = max(0, min(img_width, int(aabb.xmax)))
        y2 = max(0, min(img_height, int(aabb.ymax)))

        print(f"Bounding box {idx}: ({x1}, {y1}, {x2}, {y2})")

        if x1 < x2 and y1 < y2:
            # Crop and paste onto white background
            cropped_word = img_pil.crop((x1, y1, x2, y2))
            white_bg = Image.new("RGB", cropped_word.size, color=(255, 255, 255))
            white_bg.paste(cropped_word, (0, 0))

            # Save as JPG with sequential numbering starting at 1
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

    # Output directory to save images with bounding boxes
    output_dir = "../english"  

    # Iterate through the results and draw bounding boxes
    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        # Scale the bounding boxes back to original size
        f = loader.get_scale_factor(i)
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]

        # Get the original image (before any processing)
        img = loader.get_original_img(i)

        # Visualize and plot (optional)
        # visualize_and_plot(img, aabbs)  # Visualize before saving

        # Draw bounding boxes on the image
        draw_bounding_boxes(img, aabbs, output_dir, i)

if __name__ == '__main__':
    main()
