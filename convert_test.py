import open3d as o3d
import numpy as np
import torch

def pt_to_obj(pt_path, obj_path):
    """
    Converts a .pt file to a .obj file by reconstructing a mesh using Poisson reconstruction.
    This function assumes that the loaded tensor has a batch dimension, and the points have 6 
    values per point, but only the first 3 are used as coordinates (x, y, z).

    Args:
        pt_path (str): Path to the input .pt file.
        obj_path (str): Path to the output .obj file.
    """
    # Load the tensor
    data = torch.load(pt_path)
    print(f"Shape of the loaded .pt file: {data.shape}")

    # Handle the case where there is a batch dimension (e.g., [1, 1000, 6])
    if len(data.shape) == 3 and data.shape[0] == 1:
        data = data.squeeze(0)  # Remove the batch dimension

    # Now data should be of shape [N, 6]
    print(f"Total number of points after squeezing the batch dimension: {data.shape[0]}")

    # Check if there are enough points to proceed
    if data.shape[0] < 10:  # Arbitrarily setting 10 as the minimum number of points
        print("Not enough points in the point cloud for reconstruction. Aborting.")
        return

    # Move tensor from GPU to CPU if necessary
    if data.is_cuda:
        data = data.cpu()

    # Convert tensor to numpy array and ensure it's of type float64
    data = data.numpy().astype(np.float64)

    # Extract only the first 3 values (x, y, z) and discard the rest
    data = data[:, :3]

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    # If no normals are present, estimate them
    pcd.estimate_normals()

    # Surface reconstruction using Poisson reconstruction
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

    # Save to .obj file
    if o3d.io.write_triangle_mesh(obj_path, poisson_mesh, write_vertex_normals=False):
        print(f"Mesh data successfully saved to {obj_path}")
    else:
        print("Failed to save mesh data.")

# pt_back_to_obj_path = 'test_dataset/reconstructed_model.obj'
pt_path = 'Results\stroke_pt\stroke.pt'

pt_back_to_obj_path = 'test_dataset/cactus-2_epoch_3300.obj'

pt_to_obj(pt_path, pt_back_to_obj_path)
