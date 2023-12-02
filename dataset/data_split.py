import os
import shutil
import random

# Path to the original dataset
dataset_path = "data/train"

# Path to the new folders
cats_folder = "data/train/cats"
dogs_folder = "data/train/dogs"
val_cats_folder = "data/val/cats"
val_dogs_folder = "data/val/dogs"

# Create the new folders if they don't exist
os.makedirs(cats_folder, exist_ok=True)
os.makedirs(dogs_folder, exist_ok=True)
os.makedirs(val_cats_folder, exist_ok=True)
os.makedirs(val_dogs_folder, exist_ok=True)

# Ratio for splitting (e.g., 80% train, 20% validation)
train_ratio = 0.8

# Iterate through the files in the original dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):  # Assuming images are in JPG format
        # Extract class (cat or dog) from the filename
        class_name = filename.split('.')[0]

        # Define the source and destination paths
        source_path = os.path.join(dataset_path, filename)
        
        # Decide whether to put the image in the training or validation set
        if random.uniform(0, 1) < train_ratio:
            destination_folder = cats_folder if class_name == "cat" else dogs_folder
        else:
            destination_folder = val_cats_folder if class_name == "cat" else val_dogs_folder

        destination_path = os.path.join(destination_folder, filename)

        # Move the file to the appropriate folder
        shutil.move(source_path, destination_path)

print("Dataset has been divided into train and val sets for cats and dogs.")
