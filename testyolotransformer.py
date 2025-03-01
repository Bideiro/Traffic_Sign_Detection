import os
import yaml
import cv2
import shutil
from tqdm import tqdm
from pathlib import Path

def list_subdirectories(folder_path):
    try:
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        print(f"Subdirectories in {folder_path}: {subdirs}")  # Debugging line
        return subdirs
    except FileNotFoundError:
        print("Error: Folder not found.")
        return []

def copy_matching_files(dir1, dir2, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Traverse both directories and get filenames
    # Use os.path.splitext to remove the file extension for comparison
    dir1_files = {os.path.splitext(os.path.basename(f))[0] for f in get_all_files(dir1)}
    dir2_files = {os.path.splitext(os.path.basename(f))[0] for f in get_all_files(dir2)}

    print(f"Files in dir1 (base names): {dir1_files}")  # Debugging line
    print(f"Files in dir2 (base names): {dir2_files}")  # Debugging line

    # Compare and copy matching files
    for dirpath, _, filenames in os.walk(dir2):
        for filename in filenames:
            base_filename = os.path.splitext(filename)[0]  # Get the base name without extension
            if base_filename in dir1_files:
                # Get relative path of the file in dir2
                rel_path = os.path.relpath(dirpath, dir2)
                output_path = os.path.join(output_dir, rel_path)

                print(f"Copying {filename} from {dirpath} to {output_path}")  # Debugging line

                # Ensure the parent folder exists in the output directory
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # Copy the file from dir2 to output_dir (preserving the extension)
                file_in_dir2 = os.path.join(dirpath, filename)
                shutil.copy(file_in_dir2, output_path)
                print(f"Copied {filename} to {output_path}")


def get_all_files(dir_path):
    """Helper function to retrieve all files in a directory and its subdirectories."""
    file_paths = []
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_paths.append(full_path)
    print(f"Files in {dir_path}: {file_paths}")  # Debugging line
    return file_paths


resnet_dir = '/mnt/f/AAA Thesis Dataset Archive/test/resnet'

dataset_dir = '/mnt/f/AAA Thesis Dataset Archive/test/yolo'

output_dir = '/mnt/f/AAA Thesis Dataset Archive/test/result'
    
for i in ["/test", "/train","/valid"]:

    resnet_class_folders = list_subdirectories(resnet_dir + i)

    for x in resnet_class_folders:
        
        x = '/' + x

        # YOLO orig then resnet
        copy_matching_files(dataset_dir + i + '/images', resnet_dir + i + x, output_dir + i + x + '/images')
        copy_matching_files(dataset_dir + i + '/labels', resnet_dir + i + x, output_dir + i + x + '/labels')