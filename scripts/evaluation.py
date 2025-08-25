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

def apply_transform_to_pointcloud(pc, transform, scale=1.0):
    N = pc.shape[0]
    ones = torch.ones((N, 1), dtype=pc.dtype, device=pc.device)
    pc_hom = torch.cat([pc*scale, ones], dim=1)  # shape: (N, 4)
    transform_torch = torch.tensor(transform, dtype=pc.dtype, device=pc.device)
    pc_transformed = (transform_torch @ pc_hom.T).T[:, :3]  # (N, 3)
    return pc_transformed


def camera_center_from_c2w(c2w):
    """
    c2w: np.ndarray (4,4) o torch.Tensor (4,4)
    ritorna torch.Tensor(3,) sullo stesso device di input
    """
    if not torch.is_tensor(c2w):
        c2w = torch.from_numpy(c2w).float()
    return c2w[:3, 3]

def make_camera_axes_points(cam_t, scale=0.2, device=None, dtype=torch.float32):
    """
    Crea un piccolo set di punti che disegnano gli assi locali all'origine cam_t:
    - un punto al centro (bianco)
    - 2 punti per asse (±) così nel viewer si vede una croce lungo X,Y,Z
    Ritorna: pts (M,3), cols (M,3) in [0,1]
    """
    if device is None:
        device = cam_t.device
    cx, cy, cz = cam_t
    d = scale

    pts = torch.tensor([
        [cx,      cy,      cz],      # centro (bianco)
        [cx+d,    cy,      cz],      # +X
        [cx-d,    cy,      cz],      # -X
        [cx,      cy+d,    cz],      # +Y
        [cx,      cy-d,    cz],      # -Y
        [cx,      cy,      cz+d],    # +Z
        [cx,      cy,      cz-d],    # -Z
    ], device=device, dtype=dtype)

    cols = torch.tensor([
        [1.0, 1.0, 1.0],   # centro bianco
        [1.0, 0.0, 0.0],   # X rosso
        [1.0, 0.0, 0.0],   # X rosso
        [0.0, 1.0, 0.0],   # Y verde
        [0.0, 1.0, 0.0],   # Y verde
        [0.0, 0.0, 1.0],   # Z blu
        [0.0, 0.0, 1.0],   # Z blu
    ], device=device, dtype=dtype)

    return pts, cols

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

def get_latest_pcl_file(path_dir, obj= False):
    import re
    import os 

    max_step = -1
    latest_file = None

    # Regular expression to extract step number
    if obj:
        pattern = re.compile(r"global_pcl_obj_(\d+)\.ply")
    else:
        pattern = re.compile(r"global_pcl_(\d+)\.ply")

    for fname in os.listdir(path_dir):
        match = pattern.match(fname)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                latest_file = os.path.join(path_dir, fname)

    return latest_file

def load_ply_pointcloud(path):

    scene = trimesh.load(path, force='scene', process=False)
    vertices = []
    for geom in scene.geometry.values():
        vertices.append(geom.vertices)
    all_vertices = np.vstack(vertices)
    # print(f"{path} has {all_vertices.shape[0]} points before sampling")
    return torch.from_numpy(all_vertices).float()

def load_env_glb_pointcloud(path, num_points=400_000, device="cpu", dtype=torch.float32, apply_transform=None):
    """
    Carica un .glb/.gltf e restituisce un point cloud uniformemente campionato sulla superficie.
    - Applica automaticamente le trasformazioni dei nodi della scena.
    - Campiona per area (no bias sui soli vertici).
    - Opzionale: apply_transform (4x4) per allineare al tuo world frame (es. habitat_transform).
    """
    scene_or_mesh = trimesh.load(path, force='scene')
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Applica i transform dei nodi e concatena tutto in una sola mesh
        mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh

    if (mesh.is_empty or mesh.vertices.size == 0 or mesh.faces.size == 0):
        raise ValueError(f"[load_glb_pointcloud] Mesh vuota o senza facce: {path}")

    # Campionamento uniforme per area
    pts, _ = trimesh.sample.sample_surface(mesh, num_points)

    # (Opzionale) Applica una trasformazione extra (es. habitat_transform)
    if apply_transform is not None:
        T = np.asarray(apply_transform)
        assert T.shape == (4, 4)
        pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
        pts = (pts_h @ T.T)[:, :3]

    # Torch tensor
    pc = torch.from_numpy(pts).to(dtype=dtype)
    if device is not None:
        pc = pc.to(device)

    print(f"[load_glb_pointcloud] Loaded {path} -> {pc.shape[0]} points (uniform surface sampling)")
    return pc

def concat_mesh_from_glb(path, apply_transform=None):
    scene_or_mesh = trimesh.load(path, force='scene', process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh
    if mesh.is_empty or mesh.faces.size == 0:
        raise ValueError(f"[raycast] mesh vuota: {path}")
    if apply_transform is not None:
        T = np.asarray(apply_transform, dtype=np.float64)
        mesh.apply_transform(T)
    return mesh


def check_pcd_validity(pcd: torch.Tensor, tag="[known_env]"):
    """
    Controlla dimensioni, finitezza, bounding box e stampa info.
    """
    assert isinstance(pcd, torch.Tensor) and pcd.ndim == 2 and pcd.shape[1] == 3, f"{tag} pcd shape errata: {getattr(pcd,'shape',None)}"
    N = pcd.shape[0]
    if N < 1000:
        raise ValueError(f"{tag} troppo pochi punti: {N}")

    if not torch.isfinite(pcd).all():
        raise ValueError(f"{tag} valori non finiti nella point cloud")

    mins = torch.min(pcd, dim=0).values
    maxs = torch.max(pcd, dim=0).values
    center = (mins + maxs) / 2
    extents = (maxs - mins)
    print(f"{tag} N={N} | bbox min={mins.tolist()} max={maxs.tolist()} | size={extents.tolist()} | center={center.tolist()}")


def load_glb_pointcloud(path, num_points=100000):

    scene = trimesh.load(path, force='scene', process=False)
    vertices = []
    for geom in scene.geometry.values():
        vertices.append(geom.vertices)
    all_vertices = np.vstack(vertices)
    print(f"{path} has {all_vertices.shape[0]} points before sampling")

    if all_vertices.shape[0] > num_points:
        indices = np.random.choice(all_vertices.shape[0], num_points, replace=False)
        all_vertices = all_vertices[indices]
        print(f"Downsampled to {num_points} points")

    return torch.from_numpy(all_vertices).float()

def save_pointcloud_as_ply(pc, path, c2w=None):
    import open3d as o3d
    pc_np = pc.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)
    o3d.io.write_point_cloud(path, pcd)
    
    # --- world frame (TriangleMesh) ---
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    cam_frame = None
    if c2w is not None:
        if torch.is_tensor(c2w):
            c2w = c2w.detach().cpu().numpy()
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        cam_frame.transform(c2w)  # posiziona secondo la camera

    # --- base primitive per i punti ---
    base = o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.01)
    base.compute_vertex_normals()

    # --- merge: frame + istanze della primitive ai punti ---
    merged = o3d.geometry.TriangleMesh()
    merged += world_frame
    if cam_frame is not None:
        merged += cam_frame
    for p in pc_np:
        m = o3d.geometry.TriangleMesh(base)  # copia
        m.translate(p.tolist())
        merged += m

    merged.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, merged)
    print(f"[PLY] salvato: {path}  (punti usati={pc_np.shape[0]}")
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