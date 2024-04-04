import numpy as np
import cv2
import os


# Function to apply background noise filtering using image thresholding
def filter_background(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply adaptive thresholding to segment foreground objects from background
    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to further refine the thresholded image
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Invert the binary image to get foreground objects
    filtered_image = cv2.bitwise_and(image, image, mask=closing)

    return filtered_image


# Apply background noise filtering to all images
image_dir = 'oxford_flowers_dataset/jpg/'
image_files = sorted(os.listdir(image_dir))
filtered_image_dir = 'oxford_flowers_dataset/filtered_jpg_7/'
os.makedirs(filtered_image_dir, exist_ok=True)
count = 0
for image_file in image_files:
    image = cv2.imread(os.path.join(image_dir, image_file))
    filtered_image = filter_background(image)
    cv2.imwrite(os.path.join(filtered_image_dir, image_file), filtered_image)
    print("Background Filter in Progress --- ", count/len(image_files))
    count += 1

# Now, you can proceed with loading the filtered images and continuing with your model training as before.
