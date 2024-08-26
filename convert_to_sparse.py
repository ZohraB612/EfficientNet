import open3d as o3d
import numpy as np
import torch
import os

def obj_to_pt(obj_path, pt_path, sparsity=0.5):
    """
    Converts a .obj file to a .pt file and makes the point cloud sparser.

    Args:
        obj_path (str): Path to the input .obj file.
        pt_path (str): Path to the output .pt file.
        sparsity (float): Fraction of points to keep, must be between 0 and 1.
    """
    # Load the OBJ file
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    # Check if the mesh is empty
    if mesh.is_empty():
        print(f"The mesh {obj_path} is empty.")
        return
    
    # Extract vertices and optionally normals
    vertices = np.asarray(mesh.vertices)
    print(f"Total number of points before sparsification: {vertices.shape[0]} in {obj_path}")
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        data = np.hstack((vertices, normals))
    else:
        data = vertices
    
    # Randomly select a subset of points based on sparsity level
    total_points = data.shape[0]
    indices = np.random.choice(total_points, int(total_points * sparsity), replace=False)
    sparse_data = data[indices]
    print(f"Total number of points after sparsification: {sparse_data.shape[0]} in {obj_path}")

    # Convert numpy array to tensor and save
    tensor = torch.tensor(sparse_data, dtype=torch.float32)
    torch.save(tensor, pt_path)
    print(f"Mesh data saved to {pt_path}, kept {sparsity*100:.0f}% of points.")

def convert_folder_obj_to_pt(obj_folder, pt_folder, sparsity=0.5):
    """
    Converts all .obj files in a given folder to .pt files in a target folder.
    
    Args:
        obj_folder (str): Path to the folder containing .obj files.
        pt_folder (str): Path to the folder where .pt files will be saved.
        sparsity (float): Fraction of points to keep, must be between 0 and 1.
    """
    # Ensure the output directory exists
    os.makedirs(pt_folder, exist_ok=True)
    
    # Iterate over all .obj files in the folder
    for filename in os.listdir(obj_folder):
        if filename.endswith('.obj'):
            obj_path = os.path.join(obj_folder, filename)
            pt_path = os.path.join(pt_folder, filename.replace('.obj', '.pt'))
            obj_to_pt(obj_path, pt_path, sparsity=sparsity)

# Define paths for train and test folders
# train_obj_folder = 'test_dataset/chair/train_obj'
# test_obj_folder = 'test_dataset/chair/test_obj'

# train_pt_folder = 'test_dataset/chair/train_pt'
# test_pt_folder = 'test_dataset/chair/test_pt'

# Convert all .obj files in train and test folders to .pt
# convert_folder_obj_to_pt(train_obj_folder, train_pt_folder, sparsity=0.5)
# convert_folder_obj_to_pt(test_obj_folder, test_pt_folder, sparsity=0.5)


def pt_to_obj(pt_path, obj_path):
    """
    Converts a .pt file to a .obj file by reconstructing a mesh using Poisson reconstruction.

    Args:
        pt_path (str): Path to the input .pt file.
        obj_path (str): Path to the output .obj file.
    """
    # Load the tensor
    data = torch.load(pt_path)
    print(f"Total number of points in loaded .pt file: {data.shape[0]}")
    
    # Check if normals were included
    has_normals = data.shape[1] == 6  # Assuming 3 for vertices, 3 for normals

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3].numpy())
    if has_normals:
        pcd.normals = o3d.utility.Vector3dVector(data[:, 3:6].numpy())
    else:
        # If no normals, estimate them
        pcd.estimate_normals()

    # Surface reconstruction using Poisson reconstruction
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

    # Save to .obj file
    if o3d.io.write_triangle_mesh(obj_path, poisson_mesh, write_vertex_normals=has_normals):
        print(f"Mesh data successfully saved to {obj_path}")
    else:
        print(f"Failed to save mesh data to {obj_path}.")

def convert_folder_pt_to_obj(pt_folder, obj_folder):
    """
    Converts all .pt files in a given folder to .obj files in a target folder.
    
    Args:
        pt_folder (str): Path to the folder containing .pt files.
        obj_folder (str): Path to the folder where .obj files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(obj_folder, exist_ok=True)
    
    # Iterate over all .pt files in the folder
    for filename in os.listdir(pt_folder):
        if filename.endswith('.pt'):
            pt_path = os.path.join(pt_folder, filename)
            obj_path = os.path.join(obj_folder, filename.replace('.pt', '.obj'))
            pt_to_obj(pt_path, obj_path)

# Define paths for train and test pt folders
train_pt_folder = 'test_dataset/chair/train_pt'
test_pt_folder = 'test_dataset/chair/test_pt'

train_obj_reconstructed_folder = 'test_dataset/chair/train_obj_reconstructed'
test_obj_reconstructed_folder = 'test_dataset/chair/test_obj_reconstructed'

# Convert all .pt files in train and test folders back to .obj
convert_folder_pt_to_obj(train_pt_folder, train_obj_reconstructed_folder)
convert_folder_pt_to_obj(test_pt_folder, test_obj_reconstructed_folder)

# Example usage
# obj_path = 'test_dataset/object.obj'
# pt_path = 'test_dataset/output_model.pt'
# obj_to_pt(obj_path, pt_path, sparsity=0.01)

# pt_back_to_obj_path = 'test_dataset/reconstructed_model.obj'
# pt_path = 'Results\stroke_pt\stroke.pt'

# pt_back_to_obj_path = 'test_dataset/epoch_900.obj'

# pt_to_obj(pt_path, pt_back_to_obj_path)
