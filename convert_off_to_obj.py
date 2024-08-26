import os
import trimesh

# Path to your chair folder
base_dir = 'test_dataset/chair'

# Folders to process
folders = ['train', 'test']

# Function to convert .off files to .obj and save them in new folders
def convert_off_to_obj(source_folder, target_folder):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.off'):
                # Create the corresponding path in the target directory
                relative_path = os.path.relpath(root, source_folder)
                target_dir = os.path.join(target_folder, relative_path)
                
                # Ensure the target directory exists
                os.makedirs(target_dir, exist_ok=True)
                
                # Define source and target file paths
                off_path = os.path.join(root, file)
                obj_path = os.path.join(target_dir, os.path.splitext(file)[0] + '.obj')
                
                # Load the .off file using trimesh
                mesh = trimesh.load(off_path)
                
                # Export the mesh as .obj
                mesh.export(obj_path)
                
                print(f"Converted {off_path} to {obj_path}")

# Iterate through the 'train' and 'test' folders
for folder in folders:
    source_folder = os.path.join(base_dir, folder)
    target_folder = os.path.join(base_dir, folder + '_obj')  # New folder for .obj files
    convert_off_to_obj(source_folder, target_folder)

print("Conversion completed.")
