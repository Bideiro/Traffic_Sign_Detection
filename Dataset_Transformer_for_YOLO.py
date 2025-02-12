import os
import yaml
import shutil
from tqdm import tqdm

def delete_empty_dirs(directory):
    """Recursively delete empty directories."""
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        for dirname in dirnames:
            dir_to_check = os.path.join(dirpath, dirname)
            if not os.listdir(dir_to_check):  # Check if directory is empty
                os.rmdir(dir_to_check)

def create_directories(yaml_file, dataset_dir, output_dir):
    # Load YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # Extract class names
    classes = data.get('names', [])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create directories for each class
    class_dirs = {}
    for class_name in classes:
        class_path = os.path.join(output_dir, class_name)
        os.makedirs(os.path.join(class_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(class_path, 'texts'), exist_ok=True)
        class_dirs[class_name] = class_path

    # Create directory for multiple detections
    multi_detection_dir = os.path.join(output_dir, 'multiple_detections')
    os.makedirs(multi_detection_dir, exist_ok=True)

    # Process images and labels
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    for label_file in tqdm(label_files, desc="Processing files"):
        label_path = os.path.join(labels_dir, label_file)

        # Read label file to determine classes present
        with open(label_path, 'r') as lf:
            class_ids = {line.split()[0] for line in lf}

        class_names = [classes[int(class_id)] for class_id in class_ids]

        # Determine the output folder
        if len(class_names) == 1:
            target_class = class_names[0]
            target_dir = class_dirs[target_class]
        else:
            folder_name = "_and_".join(sorted(class_names))
            target_dir = os.path.join(multi_detection_dir, folder_name)
            os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(target_dir, 'texts'), exist_ok=True)

        # Copy image and label file to the target directory
        image_file = label_file.replace('.txt', '.jpg')  # Assuming image format is .jpg
        image_path = os.path.join(images_dir, image_file)

        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(target_dir, 'images', image_file))

        shutil.copy(label_path, os.path.join(target_dir, 'texts', label_file))
        # Delete empty directories after processing
        delete_empty_dirs(output_dir)

if __name__ == "__main__":
    
    Dataset_home_folder = "C:/Users/dei/Documents/Programming/Datasets/Dataset Archive/Dissertation.v5-latest.yolov8"  # Path to the YOLO dataset directory
    yaml_file = Dataset_home_folder + "/data.yaml"  # Path to the YOLO dataset YAML file
    output_dir = "C:/Users/dei/Documents/Programming/Datasets/YOLO_Transformed_Datasets" + "/Dissertation.v5-latest.yolov8"  # Path to the output directory

    create_directories(yaml_file, Dataset_home_folder + "/valid", output_dir + "/valid")
    create_directories(yaml_file, Dataset_home_folder + "/train", output_dir + "/train")
    create_directories(yaml_file, Dataset_home_folder + "/test", output_dir + "/test")
    

