import os
import cv2

def denoise_images_in_folder(input_folder, output_folder):
    """
    Denoise all images in a single folder and save them to the output folder.

    Parameters:
    - input_folder (str): Folder containing the images.
    - output_folder (str): Folder where denoised images will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            input_image_path = os.path.join(input_folder, file_name)
            output_image_path = os.path.join(output_folder, file_name)

            # Read the image
            image = cv2.imread(input_image_path)
            if image is None:
                print(f"Skipping invalid file: {input_image_path}")
                continue

            # Apply denoising
            denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

            # Save the denoised image
            cv2.imwrite(output_image_path, denoised_image)
            print(f"Denoised image saved: {output_image_path}")

# Example usage
input_folder = "."  # Replace with the path to your folder containing images
output_folder = "./denoised_images"  # Replace with the path for the output folder
denoise_images_in_folder(input_folder, output_folder)
