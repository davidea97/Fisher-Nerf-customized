import numpy as np
import open3d as o3d

# Load the .npz file
data = np.load("experiments/GaussianSLAM/Gibson_test_pretraind/params2000.npz")
points = data['means3D']
colors = np.clip(data['rgb_colors'], 0.0, 1.0)

# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)

# Perform Poisson reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Optionally remove low-density vertices
vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)

# Save mesh
o3d.io.write_triangle_mesh("output_mesh.ply", mesh)

# Visualize
o3d.visualization.draw_geometries([mesh])