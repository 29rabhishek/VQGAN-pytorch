import os
import random
import csv
from collections import defaultdict
from prettytable import PrettyTable

def split_data_to_csv(root_dir, train_csv, test_csv, test_size=0.2):
    """
    Splits image data into train and test CSV files while ensuring balanced class distribution.

    :param root_dir: Root directory containing class subdirectories with images.
    :param train_csv: Output CSV file path for the training data.
    :param test_csv: Output CSV file path for the testing data.
    :param test_size: Proportion of data to include in the test set (default is 0.2).
    """
    class_image_paths = defaultdict(list)

    # Collect image paths for each class
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path):
                    class_image_paths[class_name].append(file_path)

    # Split data into train and test
    train_data = []
    test_data = []

    for class_name, image_paths in class_image_paths.items():
        random.shuffle(image_paths)
        split_index = int(len(image_paths) * (1 - test_size))
        train_data.extend([(class_name, path) for path in image_paths[:split_index]])
        test_data.extend([(class_name, path) for path in image_paths[split_index:]])

    # Write train data to CSV
    with open(train_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['class_name', 'image_path'])
        writer.writerows(train_data)

    # Write test data to CSV
    with open(test_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['class_name', 'image_path'])
        writer.writerows(test_data)

def expand_training_data(train_csv, test_csv, additional_data_dir, new_train_csv, new_data_csv, max_new_images=100):
    """
    Expands the training data by adding new images from the same classes while excluding test set images.

    :param train_csv: Path to the training CSV file.
    :param test_csv: Path to the testing CSV file.
    :param additional_data_dir: Directory containing additional images for the same classes.
    :param new_train_csv: Output CSV file path for the new training data.
    :param new_data_csv: Output CSV file path for the new additional data.
    :param max_new_images: Maximum number of new images to add per class.
    """
    # Load existing train and test data
    train_data = {}
    test_data = set()

    with open(train_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            class_name, image_path = row
            if class_name not in train_data:
                train_data[class_name] = []
            train_data[class_name].append(image_path)

    with open(test_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            _, image_path = row
            test_data.add(image_path)

    new_train_data = []
    new_data = []
    added_images_count = defaultdict(int)

    # Add new images from additional data directory
    for class_name in train_data:
        class_dir = os.path.join(additional_data_dir, class_name)
        if os.path.isdir(class_dir):
            added_count = 0
            for file_name in os.listdir(class_dir):
                if added_count >= max_new_images:
                    break
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path) and file_path not in test_data:
                    new_train_data.append((class_name, file_path))
                    new_data.append((class_name, file_path))
                    added_images_count[class_name] += 1
                    added_count += 1

    # Combine existing train data with new train data
    for class_name, paths in train_data.items():
        for path in paths:
            new_train_data.append((class_name, path))

    # Write new train data to CSV
    with open(new_train_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['class_name', 'image_path'])
        writer.writerows(new_train_data)

    # Write new data to CSV
    with open(new_data_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['class_name', 'image_path'])
        writer.writerows(new_data)

    # Print the summary of added and old images in a table format
    table = PrettyTable()
    table.field_names = ["Class Name", "Old Images", "New Images"]
    for class_name, paths in train_data.items():
        old_count = len(paths)
        new_count = added_images_count[class_name]
        table.add_row([class_name, old_count, new_count])

    print(table)

# Example usage
# root_dir = "/storage/public_datasets/imagenet/ILSVRC/Data/CLS-LOC/train" 
train_csv = "train_data.csv"  # Output path for train CSV
test_csv = "test_data.csv"  # Output path for test CSV

# Additional data directory
additional_data_dir = "/storage/public_datasets/imagenet/ILSVRC/Data/CLS-LOC/train"  # Replace with the path to additional data directory
new_train_csv = "new_train_data.csv"  # Output path for new train CSV
new_data_csv = "new_data.csv"  # Output path for new additional data CSV

# split_data_to_csv(root_dir, train_csv, test_csv, test_size=0.2)
expand_training_data(train_csv, test_csv, additional_data_dir, new_train_csv, new_data_csv, max_new_images=100)
