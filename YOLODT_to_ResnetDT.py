import os
import shutil
import yaml
import cv2

def yolo_to_resnet50(yaml_path, yolo_annotations_dir, images_dir, output_dir):
    """
    Convert a YOLO dataset to a dataset format suitable for ResNet-50.
    
    Args:
        yaml_path (str): Path to the YOLO dataset YAML file.
        yolo_annotations_dir (str): Path to the directory containing YOLO annotation files (.txt).
        images_dir (str): Path to the directory containing the images.
        output_dir (str): Path to save the converted dataset (ResNet-50 format).
    """
    # Load the YOLO dataset configuration (classes, paths, etc.)
    with open(yaml_path, "r") as f:
        yolo_config = yaml.safe_load(f)
    
    # Get class names from the YAML file
    class_names = yolo_config["names"]
    print(f"Classes found: {class_names}")
    
    # Create output directories for each class
    for class_name in class_names:
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
    
    # Initialize counters for small and invalid crops
    small_crops_count = 0
    invalid_crops_count = 0
    total_crops = 0
    
    # Process each annotation file
    for annotation_file in os.listdir(yolo_annotations_dir):
        if not annotation_file.endswith(".txt"):
            continue  # Skip non-TXT files
        
        # Get the corresponding image file
        image_file = annotation_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image {image_file} not found. Skipping...")
            continue
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image {image_file}. Skipping...")
            continue
        
        # Read the annotation file
        annotation_path = os.path.join(yolo_annotations_dir, annotation_file)
        with open(annotation_path, "r") as f:
            lines = f.readlines()

        # Process each bounding box in the annotation
        for i, line in enumerate(lines):
            parts = line.strip().split()

            # Skip lines that don't have exactly 5 elements (class_id, x_center, y_center, width, height)
            if len(parts) != 5:
                print(f"Invalid line in {annotation_file}: {line.strip()}. Skipping...")
                invalid_crops_count += 1
                continue

            # Extract class ID and bounding box values
            class_id = int(parts[0])
            try:
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError as e:
                print(f"Error parsing line in {annotation_file}: {line.strip()}. Skipping...")
                invalid_crops_count += 1
                continue
                    
            # Convert YOLO normalized coordinates to pixel coordinates
            img_h, img_w = image.shape[:2]
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h
            
            # Calculate the bounding box (top-left and bottom-right corners)
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            # Ensure bounding box is within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)
            
            # Crop the image to the bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]
            
            # Skip small or invalid crops
            total_crops += 1
            if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
                print(f"Small crop for {image_file}, bbox {i}. Skipping...")
                small_crops_count += 1
                continue
            
            # Save the cropped image to the corresponding class folder
            class_name = class_names[class_id]
            output_class_dir = os.path.join(output_dir, class_name)
            cropped_image_filename = f"{os.path.splitext(image_file)[0]}_{i}.jpg"
            cropped_image_path = os.path.join(output_class_dir, cropped_image_filename)
            cv2.imwrite(cropped_image_path, cropped_image)
    
    # Summary of small and invalid crops
    print(f"\nSummary of crops:")
    print(f"Total crops processed: {total_crops}")
    print(f"Small crops skipped: {small_crops_count}")
    print(f"Invalid crops skipped: {invalid_crops_count}")
    print(f"Conversion complete. Dataset saved to: {output_dir}")

# Example usage
yaml_path = "data.yaml"  # Path to your YOLO YAML file
yolo_annotations_dir = "datasets/test/labels"  # Directory containing YOLO TXT annotation files
images_dir = "datasets/test/images"  # Directory containing image files
output_dir = "ResNet_Dataset/test"  # Where the converted dataset will be saved

yolo_to_resnet50(yaml_path, yolo_annotations_dir, images_dir, output_dir)
