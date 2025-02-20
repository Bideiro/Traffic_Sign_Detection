
"""
This python file sole purpose is to make transforming datasets more easier to the user.
Datasets will be focused on trainig with YOLO(YAML file is needed), then converted into a ResNet50v2
So that training with ResNetV2 is possible. with the ResNetV2 Dataset as a guide, convert and sort
the raw YOLO Dataset. This is done so that both datasets have the same images but also accomodate
both models for training.

Transformation Preparation guide:

    -Download all datasets first
        -it is suggested to prepare the following folders for Dataset Transformation:
            -Dataset_Archive: Contains all the Raw downloaded datasets.
            
            -Transformed_Datasets_ResNetV2: This contains the output for ResNetV2 Datasets( Step 1 ).
            -Combined_Datasets_ResNetV2: This is where the user sorts the datasets. Also can be called Main ResNetV2 Dataset. ( Step 2 & 3).
            
            -Transformed_Datasets_YOLO: This contains the output for YOLO Datasets.
            -Combined_Datasets_YOLO: This is where the user collates all the Transformed YOLO Datasets.
            
    -Make sure all datasets that will be transformed contains a YAML file <<- IMPORTANT!!
    -Have a dedicated main folder for datasets, this makes it easier for knowing which is which.

Transforming step-by-step guide:

    For ResNetV2 Model:
    1.Convert YOLO Dataset into a ResNetv2 Dataset
    2.Manually sort and tally the converted ResNetv2 Dataset into Main ResNetv2 Dataset.
    3.Collate all the transformed ResNetv2 Dataset
    
    For YOLO Model:
    4.Convert the YOLO Dataset into a similar Hierachy to ResNetV2.
    5.Get only the images from the converted ResNetV2 dataset, and its annotations.
    6.Collate all the images and annotations for YOLO.
    7.Change the images annotations using number 3.
    8.Collate all the images and annotations in a single folder for YOLO.
    
    
    Displayed below should be the folder hierarchy of both of your Main YOLO and ResNetV2 datasets.
    
    Legend:

        NOTE: Tabbed texts are subdirectories of the folder
        
        * = File
        - = Folder
    
    YOLO Folder hierarchy ( For training ):
    
    -Dataset_name
    
        -images
            *image1.jpg
            *image2.jpg
            *image3.jpg
            
        -text
            *text1.txt
            *text2.txt
            *text3.txt
            
        *data.yaml

    ResNetV2 Folder hierarchy ( For Training ):

    -Dataset_name
        -Class_name1
            *image1.jpg
            *image2.jpg
            *image3.jpg
            
        -Class_name2
            *image1.jpg
            *image2.jpg
            *image3.jpg
            
        -Class_name3
            *image1.jpg
            *image2.jpg
            *image3.jpg

"""
import os
import yaml
import cv2
import shutil
from tqdm import tqdm

""" Convert a YOLO dataset to a dataset format suitable for ResNet-50. """

def Yolo_to_ResNetV2_hierarchy(yaml_path, yolo_annotations_dir, images_dir, output_dir):
    
    with open(yaml_path, "r") as f:
        yolo_config = yaml.safe_load(f)
    
    class_names = yolo_config["names"]
    print(f"Classes found: {class_names}")
    
    for class_name in class_names:
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
    
    small_crops_count = 0
    invalid_crops_count = 0
    total_crops = 0
    
    annotation_files = [f for f in os.listdir(yolo_annotations_dir) if f.endswith(".txt")]
    
    for annotation_file in tqdm(annotation_files, desc="Processing Annotations", unit="file"):
        image_file = annotation_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image {image_file} not found. Skipping...")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image {image_file}. Skipping...")
            continue
        
        annotation_path = os.path.join(yolo_annotations_dir, annotation_file)
        with open(annotation_path, "r") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid line in {annotation_file}: {line.strip()}. Skipping...")
                invalid_crops_count += 1
                continue
            
            class_id = int(parts[0])
            try:
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                print(f"Error parsing line in {annotation_file}: {line.strip()}. Skipping...")
                invalid_crops_count += 1
                continue
                
            img_h, img_w = image.shape[:2]
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h
            
            x_min = max(0, int(x_center - width / 2))
            y_min = max(0, int(y_center - height / 2))
            x_max = min(img_w, int(x_center + width / 2))
            y_max = min(img_h, int(y_center + height / 2))
            
            cropped_image = image[y_min:y_max, x_min:x_max]
            
            total_crops += 1
            if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
                small_crops_count += 1
                continue
            
            class_name = class_names[class_id]
            output_class_dir = os.path.join(output_dir, class_name)
            cropped_image_filename = f"{os.path.splitext(image_file)[0]}_{i}.jpg"
            cropped_image_path = os.path.join(output_class_dir, cropped_image_filename)
            cv2.imwrite(cropped_image_path, cropped_image)
    
    # Remove empty directories
    for class_name in class_names:
        class_output_dir = os.path.join(output_dir, class_name)
        if not os.listdir(class_output_dir):
            os.rmdir(class_output_dir)
    
def Yolo_to_ResNetV2_hierachy_YOLO(dataset_dir, output_dir):
    
    yaml_file = dataset_dir + '/data.yaml'
    
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
            os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
            shutil.copy(image_path, os.path.join(target_dir, 'images', image_file))

        # Ensure the texts directory exists before copying labels
        os.makedirs(os.path.join(target_dir, 'texts'), exist_ok=True)
        
        shutil.copy(label_path, os.path.join(target_dir, 'texts', label_file))
        # Delete empty directories after processing
        delete_empty_dirs(output_dir)

def delete_empty_dirs(directory):
    """Recursively delete empty directories."""
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        for dirname in dirnames:
            dir_to_check = os.path.join(dirpath, dirname)
            if not os.listdir(dir_to_check):  # Check if directory is empty
                os.rmdir(dir_to_check)

def update_Classes(labels_dir, new_class, output_dir):
    """
    Update all class numbers in YOLO annotation files to a new class number and save to a new directory.

    Args:
        labels_dir (str): Path to the directory containing label files.
        new_class (int): The new class number.
        output_dir (str): Path to the directory where updated files will be saved.
    """
    multi_class_files = []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all subdirectories named 'texts'
    text_dirs = []
    for root, dirs, _ in os.walk(labels_dir):
        if 'texts' in dirs:
            text_dirs.append(os.path.join(root, 'texts'))

    # Collect all .txt files in 'texts' directories
    all_files = []
    for text_dir in text_dirs:
        for root, _, files in os.walk(text_dir):
            for file in files:
                if file.endswith(".txt"):
                    all_files.append(os.path.join(root, file))

    # Process files with a progress bar
    for file_path in tqdm(all_files, desc="Updating labels"):
        with open(file_path, "r") as f:
            lines = f.readlines()

        updated_lines = []
        class_set = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                class_set.add(parts[0])
                parts[0] = str(new_class)
                updated_lines.append(" ".join(parts))

        # Determine output file path
        relative_path = os.path.relpath(file_path, labels_dir)
        output_file_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Write updated content to the new file
        with open(output_file_path, "w") as f:
            f.write("\n".join(updated_lines))

        # Track files with multiple unique classes
        if len(class_set) > 1:
            multi_class_files.append(file_path)

    # Output files with multiple classes
    if multi_class_files:
        print("Files with multiple classes:")
        for file in multi_class_files:
            print(file)


def YOLO_to_Resnet(dataset_dir, output_dir):
    
    yaml_path = dataset_dir + "/data.yaml"
    
    try:
        Yolo_to_ResNetV2_hierarchy(yaml_path, dataset_dir + "/test/labels", dataset_dir + "/test/images", output_dir + "/test")
        
        Yolo_to_ResNetV2_hierarchy(yaml_path, dataset_dir + "/train/labels", dataset_dir + "/train/images", output_dir + "/train")
        
        Yolo_to_ResNetV2_hierarchy(yaml_path, dataset_dir + "/valid/labels", dataset_dir + "/valid/images", output_dir+ "/valid")
        
    except Exception as e:
        print(e)
        
def YOLO_to_Transformed_YOLO(dataset_dir, output_dir):
    
    Yolo_to_ResNetV2_hierachy_YOLO(dataset_dir + "/test/images", output_dir + "/test")
    
    Yolo_to_ResNetV2_hierachy_YOLO(dataset_dir + "/train/images", output_dir + "/train")
    
    Yolo_to_ResNetV2_hierachy_YOLO(dataset_dir + "/valid/images",  output_dir+ "/valid")
    
    
if __name__ == "__main__":
    
    dataset_dir = input("Set dataset location: ")
    output_dir = input("Set Ouput location: ")
    
    print("""
        Pick an action:
        [1] Convert a YOLO Dataset into a ResNetV2 Dataset.
        [2] Convert a YOLO Dataset into a similar Hierachy to ResNetV2.
        [3] Rename YOLO Categories.
        [4] Delete empty directories.
        """)
    
    x = input("Choice? :")
    
    """for a predefined choice change this if else and just press enter"""
    
    if x == '\n':
        x = ''
        
    if x == '1':
        YOLO_to_Resnet(dataset_dir,output_dir)
    elif x == '2':
        YOLO_to_Transformed_YOLO(dataset_dir, output_dir)
    elif x == '3':
        
        labels_dir =input("Enter labels directory:")
        
        new_class = input("Input new class number:")
        
        output_dir = input("Enter output directory:")
        
        update_Classes(labels_dir, new_class, output_dir)
    elif x == '4':
        dir = input("Input directory:")
        delete_empty_dirs(dir)
