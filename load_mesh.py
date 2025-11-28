import open3d as o3d

# Load mesh as point cloud
mesh = o3d.io.read_triangle_mesh("/Users/denismbeyakola/Desktop/space_robotics/meshes/terrain_mesh2.obj")
pcd = mesh.sample_points_uniformly(number_of_points=50000)  # convert to point cloud

# Estimate normals (mandatory for Poisson)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))

# Orient normals consistently (optional but improves reconstruction)
pcd.orient_normals_consistent_tangent_plane(100)

# Poisson reconstruction
mesh_reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Crop mesh to original bounding box
bbox = pcd.get_axis_aligned_bounding_box()
mesh_reconstructed = mesh_reconstructed.crop(bbox)

# Save
o3d.io.write_triangle_mesh("terrain_mesh_fixed.obj", mesh_reconstructed)
print("Saved as terrain_mesh_fixed.obj")
