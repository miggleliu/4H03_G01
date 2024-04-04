import cv2
import os
from sklearn.decomposition import PCA
import joblib


PCA_SKIP = False
TRAIN_SKIP = False
PCA_THRESHOLD1 = 0.98 # 702 - 98% toy; # 184 PC - 80%; 693 PC - 90%
PCA_THRESHOLD2 = 702
PCA_THRESHOLD = PCA_THRESHOLD2
VERBOSE = 0

output_dir = 'oxford_flowers_dataset/postPCA_jpg_098_toy'

##########################################################################################
##########################################################################################
# Step 1: Transform Images and Reduce Dimensionality using PCA
print(" >>> Step 1: Transform Images and Reduce Dimensionality using PCA <<<")

if PCA_SKIP:
    print("PCA skipped successfully. Now use pre-saved post-PCA dataset!")
else:
    # Load images
    print("Loading image dataset ...")
    # image_dir = 'oxford_flowers_dataset/jpg/'
    image_dir = 'oxford_flowers_dataset/filtered_jpg_7'
    image_files = sorted(os.listdir(image_dir))[:946]
    images = []
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_dir, image_file), cv2.IMREAD_GRAYSCALE)  # Read images as grayscale
        image = cv2.resize(image, (200, 200))  # Resize images to a common size
        images.append(image.flatten())  # Reshape images into column vectors

    # Apply PCA for dimensionality reduction
    print(f"Applying PCA to the dataset with threshold of {PCA_THRESHOLD} (will take a while) ...")
    pca = PCA(n_components=PCA_THRESHOLD)  # Adjusted to capture 90% of variance
    images_pca = pca.fit_transform(images)
    # Print the number of principal components used
    print("Number of principal components used:", pca.n_components_)

    # Save the trained PCA model to a file
    os.makedirs('models', exist_ok=True)
    joblib.dump(pca, 'models/pca_model_098_toy.pkl')
    print("PCA model saved successfully.")

    # Create a folder for post-PCA images if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the images after PCA with original indices
    for i, image_pca in enumerate(images_pca):
        # Reshape PCA transformed image back to its original shape
        image_pca_reshaped = pca.inverse_transform(image_pca).reshape(200, 200)

        # Save the image to the output directory with the original index
        output_file = os.path.join(output_dir, f'image_{i}.png')
        cv2.imwrite(output_file, image_pca_reshaped)

    print("Post-PCA images saved successfully.")
