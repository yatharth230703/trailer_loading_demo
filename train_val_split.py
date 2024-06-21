import os
import random
import shutil

# Directories
base_dir = r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\yolo_finetune_data'
image_dir = os.path.join(base_dir, 'chair_dataset')
label_dir = os.path.join(base_dir, 'chair_label_final')

train_image_dir = os.path.join(base_dir, 'images', 'train')
val_image_dir = os.path.join(base_dir, 'images', 'val')
train_label_dir = os.path.join(base_dir, 'labels', 'train')
val_label_dir = os.path.join(base_dir, 'labels', 'val')

# Ensure the output directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle and split the dataset
random.seed(42)  # For reproducibility
random.shuffle(image_files)
split_index = int(0.8 * len(image_files))

train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Function to copy files
def copy_files(file_list, source_image_dir, source_label_dir, dest_image_dir, dest_label_dir):
    for image_file in file_list:
        # Copy image file
        shutil.copy(os.path.join(source_image_dir, image_file), os.path.join(dest_image_dir, image_file))
        
        # Copy label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        if os.path.exists(os.path.join(source_label_dir, label_file)):
            shutil.copy(os.path.join(source_label_dir, label_file), os.path.join(dest_label_dir, label_file))

# Copy training files
copy_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)

# Copy validation files
copy_files(val_files, image_dir, label_dir, val_image_dir, val_label_dir)

print("Dataset split into training and validation sets.")
