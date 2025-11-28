import numpy as np
import open3d as o3d
from PIL import Image

# -----------------------------
# Load height map
# -----------------------------
height_map = np.array(Image.open("terrain.png")).astype(np.float32)

# Filter out zero-height pixels
mask = height_map != 0
filtered_height = height_map[mask]

# Normalize heights to 0-1
min_h, max_h = filtered_height.min(), filtered_height.max()
normalized_height = (filtered_height - min_h) / (max_h - min_h)

H, W = height_map.shape
x = np.arange(W)
y = np.arange(H)
xx, yy = np.meshgrid(x, y)

# Flatten arrays and apply mask
points = np.column_stack((xx.flatten()[mask.flatten()],
                          yy.flatten()[mask.flatten()],
                          normalized_height))

# -----------------------------
# Create point cloud
# -----------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Estimate normals (needed for meshing)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

# -----------------------------
# Poisson surface reconstruction (mesh)
# -----------------------------
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
mesh.compute_vertex_normals()

# Crop mesh to point cloud bounds (optional)
bbox = pcd.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

# Simplify mesh to reduce complexity for MuJoCo
mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=5000)

# -----------------------------
# Optional: scale Z to exaggerate bumps
# -----------------------------
vertices = np.asarray(mesh.vertices)
vertices[:, 2] *= 5  # adjust factor to your terrain scale
mesh.vertices = o3d.utility.Vector3dVector(vertices)

# -----------------------------
# Save outputs
# -----------------------------
o3d.io.write_point_cloud("lunar_points_normalized.ply", pcd)
o3d.io.write_triangle_mesh("lunar_terrain.obj", mesh)

# Visualize mesh
o3d.visualization.draw_geometries([mesh])



# import numpy as np
# import open3d as o3d
# from PIL import Image

# # Load height map
# height_map = np.array(Image.open("terrain.png")).astype(np.float32)

# # Filter out zero-height pixels
# mask = height_map != 0
# filtered_height = height_map[mask]

# # Normalize heights to 0-1
# min_h, max_h = filtered_height.min(), filtered_height.max()
# normalized_height = (filtered_height - min_h) / (max_h - min_h)

# H, W = height_map.shape
# x = np.arange(W)
# y = np.arange(H)
# xx, yy = np.meshgrid(x, y)

# # Flatten arrays
# xx_flat = xx.flatten()
# yy_flat = yy.flatten()
# zz_flat = height_map.flatten()

# # Apply mask and normalized height
# points = np.column_stack((xx_flat[mask.flatten()], yy_flat[mask.flatten()], normalized_height))

# # Create point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# # Set all points to gray color
# gray = np.full((points.shape[0], 3), 0.5)  # RGB = 0.5 gray
# pcd.colors = o3d.utility.Vector3dVector(gray)

# # Estimate normals (optional)
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))

# # Save and visualize
# o3d.io.write_point_cloud("lunar_points_gray.ply", pcd)
# o3d.visualization.draw_geometries([pcd])
