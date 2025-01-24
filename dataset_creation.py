import os
import random
from sklearn.model_selection import train_test_split

def stratified_split_image_dataset(original_dir, output_dir, test_size=0.1, random_seed=42):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Gather all image paths and their corresponding class labels
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(original_dir))
    
    for class_name in class_names:
        class_dir = os.path.join(original_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if os.path.isfile(image_path):
                    image_paths.append(image_path)
                    labels.append(class_name)

    # Perform stratified split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=random_seed
    )

    # Write train.txt and test.txt
    train_file = os.path.join(output_dir, "5k_imagenet_train.txt")
    test_file = os.path.join(output_dir, "5k_imagenet_test.txt")

    with open(train_file, "w") as f:
        for path, label in zip(train_paths, train_labels):
            f.write(f"{path} {label}\n")

    with open(test_file, "w") as f:
        for path, label in zip(test_paths, test_labels):
            f.write(f"{path} {label}\n")

    print(f"Train and test splits saved in {output_dir}.")

# Example usage
original_dir = "vqgan_eeg_imagenet_5k"  # Replace with your dataset path
output_dir = "txt files"      # Replace with your desired output path
stratified_split_image_dataset(original_dir, output_dir)
