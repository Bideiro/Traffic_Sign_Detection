import os
from tqdm import tqdm
import yaml
import cv2

def yolo_to_resnet50(yaml_path, yolo_annotations_dir, images_dir, output_dir):
    """
    Convert a YOLO dataset to a dataset format suitable for ResNet-50.
    """
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
            
# Example usage
Dataset_home_folder = "C:/Users/dei/Documents/Programming/Datasets/Dataset Archive/TrafficSign.v6i.yolov8 (1)"
output_dir = "C:/Users/dei/Documents/Programming/Datasets/Transformed_Datasets" + "/TrafficSign.v6i.yolov8 (1)" # Where the converted dataset will be saved


yolo_to_resnet50(Dataset_home_folder + "/data.yaml", Dataset_home_folder + "/test/labels", Dataset_home_folder + "/test/images", output_dir + "/test")

yolo_to_resnet50(Dataset_home_folder + "/data.yaml", Dataset_home_folder + "/train/labels", Dataset_home_folder + "/train/images", output_dir + "/train")

