import numpy as np
import cv2
import scipy.io
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras import layers, models, callbacks, regularizers
from keras.utils import to_categorical


# from keras.models import load_model
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import joblib
# import tensorflow as tf

PCA_SKIP = True
TRAIN_SKIP = False
VERBOSE = 0

postPCA_image_dir = 'oxford_flowers_dataset/postPCA_jpg_098_toy'
postPCA_image_dir = 'oxford_flowers_dataset/filtered_jpg_7'

# Load labels from .mat file
print("Loading labels ...")
mat_file = 'oxford_flowers_dataset/imagelabels.mat'
labels_data = scipy.io.loadmat(mat_file)
labels = labels_data['labels']
labels = labels.flatten()
labels = [each - 1 for each in labels]
print("Labels loaded successfully.")
labels = labels[:946]
print(labels)
# Create a dictionary to map each unique number to an integer from 1 to 5
mapping = {num: i for i, num in enumerate(sorted(set(labels)))}
# Replace random numbers with mapped integers
labels = [mapping[num] for num in labels]
print(labels)


##########################################################################################
##########################################################################################
# Step 2: Train Model and Output Training Error
print(">>> Step 2: Train Model and Output Training Error <<<")

# Load images from the 'postPCA_dataset' folder in color (RGB)
postPCA_image_files = sorted(os.listdir(postPCA_image_dir))[:946]
images_pca = []
for image_file in postPCA_image_files:
    image = cv2.imread(os.path.join(postPCA_image_dir, image_file))  # Read images in color (RGB)
    image = cv2.resize(image, (200, 200))  # Resize images to a common size
    images_pca.append(image)  # Append color images
images_pca = np.array(images_pca) / 255.0  # normalize

# Train a classifier
print("Training model using CNN ...")

# Split the dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images_pca, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  # Further split training set into training and validation sets

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(200, 200, 3)),  # Change input shape to (200, 200, 3)
    layers.MaxPooling2D((2, 2), strides=2),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2), strides=2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping callback
early_stopping_callback = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(np.array(X_train), np.array(y_train),
                    epochs=100,
                    batch_size=32,
                    validation_data=(np.array(X_val), np.array(y_val)),
                    callbacks=[early_stopping_callback])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test))
print("Test Accuracy:", test_acc)

print("Model trained successfully.")
# Save the trained model
output_dir = 'model'
os.makedirs(output_dir, exist_ok=True)
model_filename = 'flower_classification_model_rgb.keras'
model.save(model_filename)
print("Model saved to:", model_filename)
