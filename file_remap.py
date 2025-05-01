import os

def standardize_folder_names(root_dir):
    for folder in os.listdir(root_dir):
        old_path = os.path.join(root_dir, folder)
        if os.path.isdir(old_path):
            new_name = folder.strip().lower().replace(" ", "_")
            new_path = os.path.join(root_dir, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {folder} -> {new_name}")

# Run this for each of the 3 sets:
for split in ['train', 'test', 'validation']:
    dir_path = os.path.join('vegetable_images', split)
    standardize_folder_names(dir_path)
