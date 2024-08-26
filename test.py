import open3d as o3d

def obj_to_pointcloud(obj_path, output_path):
    # Load the OBJ file
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    # Check if the mesh is empty
    if mesh.is_empty():
        print("The mesh is empty.")
        return
    
    # Print out the number of vertices to verify the mesh has data
    print(f"Mesh has {len(mesh.vertices)} vertices.")

    # Compute vertex normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Convert to point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices

    # Include colors and normals if present
    if mesh.has_vertex_colors():
        point_cloud.colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        point_cloud.normals = mesh.vertex_normals

    # Save the point cloud
    if o3d.io.write_point_cloud(output_path, point_cloud):
        print(f"Point cloud successfully saved to {output_path}")
    else:
        print("Failed to save point cloud.")

# Update paths accordingly
obj_path = 'test_dataset/object.obj'
output_path = 'test_dataset/output_model.ply'
obj_to_pointcloud(obj_path, output_path)

