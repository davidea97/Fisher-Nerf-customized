import open3d as o3d
import trimesh
import torch
import numpy as np
from tqdm import tqdm
import argparse


habitat_transform = np.array([
    [1.,  0.,  0., 0.],
    [0., -1.,  0., 0.],
    [0.,  0., -1., 0.],
    [0.,  0.,  0., 1.]
])

rotation_90_x = np.array([
    [1., 0.,  0., 0.],
    [0., 0., -1., 0.],
    [0., 1.,  0., 0.],
    [0., 0.,  0., 1.]
])

def apply_transform_to_pointcloud(pc, transform):
    N = pc.shape[0]
    ones = torch.ones((N, 1), dtype=pc.dtype, device=pc.device)
    pc_hom = torch.cat([pc, ones], dim=1)  # shape: (N, 4)
    transform_torch = torch.tensor(transform, dtype=pc.dtype, device=pc.device)
    pc_transformed = (transform_torch @ pc_hom.T).T[:, :3]  # (N, 3)
    return pc_transformed

def random_sample_pc(pc, num_samples):
    n_points = pc.shape[0]
    if n_points <= num_samples:
        return pc
    else:
        indices = torch.randperm(n_points)[:num_samples]
        return pc[indices]
        
def find_nearest_points_distances(pc1, pc2):
    distances = torch.cdist(pc1, pc2, p=2)
    min_distances, _ = torch.min(distances, dim=1)
    return min_distances

def calculate_coverage_percentage(pc1, pc2, threshold=0.05, weight=1):
    if len(pc2) == 0:
        return 0.
    else:
        sampled_pc2 = random_sample_pc(pc2, int(len(pc1)*weight))
        nearest_distances = find_nearest_points_distances(pc1, sampled_pc2)
        similar_points = (nearest_distances < threshold).float().mean()
        return similar_points.item()

# def load_ply_pointcloud(path):
#     pcd = o3d.io.read_point_cloud(path)
#     print(f"{path} has {len(pcd.points)} points")
#     return torch.from_numpy(np.asarray(pcd.points)).float()

def load_ply_pointcloud(path):

    scene = trimesh.load(path, force='scene', process=False)
    vertices = []
    for geom in scene.geometry.values():
        vertices.append(geom.vertices)
    all_vertices = np.vstack(vertices)
    # print(f"{path} has {all_vertices.shape[0]} points before sampling")
    return torch.from_numpy(all_vertices).float()

def load_glb_pointcloud(path, num_points=100000):

    scene = trimesh.load(path, force='scene', process=False)
    vertices = []
    for geom in scene.geometry.values():
        vertices.append(geom.vertices)
    all_vertices = np.vstack(vertices)
    # print(f"{path} has {all_vertices.shape[0]} points before sampling")

    # if all_vertices.shape[0] > num_points:
    #     indices = np.random.choice(all_vertices.shape[0], num_points, replace=False)
    #     all_vertices = all_vertices[indices]
        # print(f"Downsampled to {num_points} points")

    return torch.from_numpy(all_vertices).float()

def save_pointcloud_as_ply(pc, path):
    import open3d as o3d
    pc_np = pc.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)
    o3d.io.write_point_cloud(path, pcd)
    # print(f"Saved point cloud to {path}")

def main(glb_path, ply_path, scene=None):
    pc1 = load_glb_pointcloud(glb_path)
    pc2 = load_ply_pointcloud(ply_path)

    # save_pointcloud_as_ply(pc2, "original_ply.ply")
    pc2_rotated = apply_transform_to_pointcloud(pc2, habitat_transform)
    # save_pointcloud_as_ply(pc2_rotated, "rotated_ply.ply")

    # save_pointcloud_as_ply(pc1, "original_glb.ply")
    pc1_rotated = apply_transform_to_pointcloud(pc1, rotation_90_x)
    # save_pointcloud_as_ply(pc1_rotated, "original_pcl.ply")

    coverage = calculate_coverage_percentage(pc1_rotated, pc2_rotated)
    print(f"#### Scene : {scene} ####")
    print(f"Coverage Percentage: {coverage * 100:.2f}%")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Path to the .glb file")
    args = parser.parse_args()
    # Replace with your actual paths
    glb_file = f"../../data/versioned_data/gibson/{args.scene}.glb"
    ply_file = f"../experiments/GaussianSLAM/{args.scene}-eccv_reproduce/pointcloud/global_pcl_1000.ply"
    # ply_file = f"../experiments/GaussianSLAM/{args.scene}-eccv_reproduce/pointcloud/global_pcl_1999.ply"
    main(glb_file, ply_file, args.scene)