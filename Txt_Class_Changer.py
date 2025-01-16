import os
from tqdm import tqdm

def update_all_classes_in_yolo_labels(labels_dir, new_class, output_dir):
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

    # Collect all .txt files
    all_files = []
    for root, _, files in os.walk(labels_dir):
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

if __name__ == "__main__":
    labels_dir = input("Enter the path to the labels directory: ").strip()
    new_class = int(input("Enter the new class number: ").strip())
    output_dir = "Combined_Dataset/train/Crosswalk/labels"

    update_all_classes_in_yolo_labels(labels_dir, new_class, output_dir)
    print(f"Class numbers updated successfully! Updated files are saved in '{output_dir}'.")
