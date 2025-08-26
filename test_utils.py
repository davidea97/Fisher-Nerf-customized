
import numpy as np
import quaternion
import datasets.util.utils as utils
import datasets.util.map_utils as map_utils
import torch
import os
import metrics
from scipy.spatial.transform import Rotation as SciR
import habitat_sim
from habitat.utils.visualizations import maps
from sklearn.metrics import confusion_matrix
from scripts.evaluation import save_pointcloud_as_ply
import cv2
import trimesh
from scipy.spatial import cKDTree
import yaml

def yaml_safe_load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)

def yaml_safe_dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def draw_map(sim, height, meters_per_pixel = 0.1, use_sim=False, map_res=768):
    """
    Draw the topdown map of the environment

    Example Usage:
        set_agent_state(test_ds.sim.sim, agent_height)

    Args:
        sim (habitat_sim.Simulator): habitat simulator object
        height (float): height of the agent
        meters_per_pixel (float): meters per pixel
    """
    if use_sim:
        hablab_topdown_map = maps.get_topdown_map_from_sim(
            sim,
            map_resolution = map_res,
            draw_border = True,
            meters_per_pixel = None,
            agent_id = 0,
        )
    else:
        hablab_topdown_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
    
    recolor_map = np.array(
        [[255, 255, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8
    )
    colored_map = recolor_map[hablab_topdown_map]

    return colored_map, hablab_topdown_map

def set_agent_state(sim, c2w: np.array):
    """
    Set the agent state in the simulator

    Example Usage:
        set_agent_state(test_ds.sim.sim, np.concatenate([pos, quat]))

    Args:
        sim (habitat_sim.Simulator): habitat simulator object
        c2w (np.array): camera to world transformation
                        if is 4x4, then it is a homogenous matrix
                        if is 7, then it is a position and quaternion (x, y, z, qw, qx, qy, qz)
    """
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.cpu().numpy()

    agent_state=sim.get_agent_state()

    if c2w.size == 16:
        # set x,z position
        agent_state.position[0] = c2w[0, 3]
        agent_state.position[2] = c2w[2, 3]
        agent_state.sensor_states["rgb"].position[0] = c2w[0, 3]
        agent_state.sensor_states["rgb"].position[2] = c2w[2, 3]
        agent_state.sensor_states["depth"].position[0] = c2w[0, 3]
        agent_state.sensor_states["depth"].position[2] = c2w[2, 3]

        rotation = c2w[:3, :3]
        # since the the habitat is x(right), y(up), z(backward),
        # we flip the 
        rotation[:3, 1] *= -1
        rotation[:3, 2] *= -1
        rot_quat = SciR.from_matrix(rotation).as_quat()
        agent_state.rotation.y = rot_quat[1]
        agent_state.rotation.w = rot_quat[3]
        agent_state.sensor_states["rgb"].rotation.y = rot_quat[1]
        agent_state.sensor_states["rgb"].rotation.w = rot_quat[3]
        agent_state.sensor_states["depth"].rotation.y = rot_quat[1]
        agent_state.sensor_states["depth"].rotation.w = rot_quat[3]
    
    elif c2w.size == 7:
        pos = c2w[:3]
        quat = c2w[3:]
        
        # set x,z position
        agent_state.position[0] = pos[0]
        agent_state.position[2] = pos[2]
        agent_state.sensor_states["rgb"].position[0] = pos[0]
        agent_state.sensor_states["rgb"].position[2] = pos[2]
        agent_state.sensor_states["depth"].position[0] = pos[0]
        agent_state.sensor_states["depth"].position[2] = pos[2]

        agent_state.rotation.y = quat[2]
        agent_state.rotation.w = quat[0]
        agent_state.sensor_states["rgb"].rotation.y = quat[2]
        agent_state.sensor_states["rgb"].rotation.w = quat[0]
        agent_state.sensor_states["depth"].rotation.y = quat[2]
        agent_state.sensor_states["depth"].rotation.w = quat[0]
    
    # manually place agent here
    sim.agents[0].set_state(agent_state)

def get_latest_model(save_dir):
    checkpoint_list = []
    for dirpath, _, filenames in os.walk(save_dir):
        for filename in filenames:
            if filename.endswith('.pt') or filename.endswith('.pth'):
                checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
    checkpoint_list = sorted(checkpoint_list)
    latest_checkpoint =  None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
    return latest_checkpoint


def load_model(models, checkpoint_file):
    # Load the latest checkpoint
    checkpoint = torch.load(checkpoint_file)
    for model in models:
        if model in checkpoint['models']:
            models[model].load_state_dict(checkpoint['models'][model])
        else:
            raise Exception("Missing model in checkpoint: {}".format(model))
    return models


def get_2d_pose(position, rotation=None):
    # position is 3-element list
    # rotation is 4-element list representing a quaternion
    position = np.asarray(position, dtype=np.float32)
    x = -position[2]
    y = -position[0]
    height = position[1]

    if rotation is not None:
        rotation = np.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
        axis = quaternion.as_euler_angles(rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        pose = x, y, o
    else:
        pose = x, y, 0.0
    return pose, height



def get_coord_pose(sg, rel_pose, init_pose, grid_dim, cell_size, device):
    # Create a grid where the starting location is always at the center looking upwards (like the ground-projected grids)
    # Then use the spatial transformer to move that location at the right place
    if isinstance(init_pose, list) or isinstance(init_pose, tuple):
        init_pose = torch.tensor(init_pose).unsqueeze(0)
    else:
        init_pose = init_pose.unsqueeze(0)
    init_pose = init_pose.to(device)

    zero_pose = torch.tensor([[0., 0., 0.]]).to(device)

    zero_coords = map_utils.discretize_coords(x=zero_pose[:,0],
                                            z=zero_pose[:,1],
                                            grid_dim=(grid_dim, grid_dim),
                                            cell_size=cell_size)

    pose_grid = torch.zeros((1, 1, grid_dim, grid_dim), dtype=torch.float32).to(device)
    pose_grid[0,0,zero_coords[0,0], zero_coords[0,1]] = 1

    pose_grid_transf = sg.spatialTransformer(grid=pose_grid, pose=rel_pose, abs_pose=init_pose)
    
    pose_grid_transf = pose_grid_transf.squeeze(0).squeeze(0)
    inds = utils.unravel_index(pose_grid_transf.argmax(), pose_grid_transf.shape)

    pose_coord = torch.zeros((1, 1, 2), dtype=torch.int64).to(device)
    pose_coord[0,0,0] = inds[1] # inds is y,x
    pose_coord[0,0,1] = inds[0]
    return pose_coord


def decide_stop(dist, stop_dist):
    if dist <= stop_dist:
        return True
    else:
        return False


# Return success, SPL, soft_SPL, distance_to_goal measures
def get_metrics(sim,
                episode_goal_positions,
                success_distance,
                start_end_episode_distance,
                agent_episode_distance,
                stop_signal):

    curr_pos = sim.get_agent_state().position

    # returns distance to the closest goal position
    distance_to_goal = sim.geodesic_distance(curr_pos, episode_goal_positions)

    if distance_to_goal <= success_distance and stop_signal:
        success = 1.0
    else:
        success = 0.0

    spl = success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )

    ep_soft_success = max(0, (1 - distance_to_goal / start_end_episode_distance) )
    soft_spl = ep_soft_success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )

    nav_metrics = {'distance_to_goal':distance_to_goal,
               'success':success,
               'spl':spl,
               'softspl':soft_spl}
    return nav_metrics


def get_map_metrics(init_gt_map, sg, cell_size, spatial_labels):

    occ_grid = sg.occ_grid.unsqueeze(0).cpu().numpy()
    proj_grid = sg.proj_grid.unsqueeze(0).cpu().numpy()
    init_gt_map = init_gt_map.cpu().numpy()
    sqm = cell_size*cell_size # square meters (0.05m -> 0.0025m2)

    # OccAnt defined map accuracy over the built map (occupied, free), i.e. need to remove 'unexplored' locations from predicted and gt maps
    explored_pred_grid = map_utils.get_explored_grid(sg.occ_grid, thresh=0.34).squeeze(0).cpu().numpy()
    explored_pred_grid = explored_pred_grid.flatten().astype(int)
    inds_for_map_acc = np.nonzero(explored_pred_grid)[0]
    
    occ_grid = np.argmax(occ_grid, axis=2).squeeze(0) # B x T x grid_dim x grid_dim
    occ_grid_for_map_acc = occ_grid.flatten()[inds_for_map_acc]
    init_gt_map_for_map_acc = init_gt_map.flatten()[inds_for_map_acc]

    ## Acc, IoU, F1 ##
    current_confusion_matrix_spatial = confusion_matrix(y_true=init_gt_map_for_map_acc, y_pred=occ_grid_for_map_acc, labels=[0,1,2])
    current_confusion_matrix_spatial = torch.tensor(current_confusion_matrix_spatial)
    mAcc_sp = metrics.overall_pixel_accuracy(current_confusion_matrix_spatial)
    class_mAcc_sp, per_class_Acc = metrics.per_class_pixel_accuracy(current_confusion_matrix_spatial)
    mIoU_sp, per_class_IoU = metrics.jaccard_index(current_confusion_matrix_spatial)
    mF1_sp, per_class_F1 = metrics.F1_Score(current_confusion_matrix_spatial)

    ## Get map accuracy in m2 (as defined in Occupancy Anticipation paper)
    correct_cells = mAcc_sp.item() * len(inds_for_map_acc)
    map_acc_m2 = correct_cells * sqm

    ## Coverage ##
    proj_grid = np.argmax(proj_grid, axis=2).squeeze(0)
    binary_inds = np.where(proj_grid.flatten()!=0, 1, 0)
    inds = np.nonzero(binary_inds)[0]
    n_explored_tiles = len(inds)
    cov = n_explored_tiles*sqm

    ## Get coverage percentage ##
    # find traversable+occupied area in gt map
    binary_inds = np.where(init_gt_map!=0, 1, 0) # 1 x H x W
    inds = np.nonzero(binary_inds)[0]
    n_total_tiles = len(inds)
    cov_per = n_explored_tiles / n_total_tiles
    
    map_metrics = {'map_accuracy':mAcc_sp.item(),
                   'map_accuracy_m2':map_acc_m2,
                   'iou': mIoU_sp.item(),
                   'f1': mF1_sp.item(),
                   'cov': cov,
                   'cov_per': cov_per}
    return map_metrics

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    



# DAVIDE
@torch.no_grad()
def render_expected_depth_from_pcd(pcd_world: torch.Tensor, c2w_t: torch.Tensor, K_t: torch.Tensor, img_hw, z_forward="-Z"):
    H, W = img_hw
    device, dtype = pcd_world.device, pcd_world.dtype
    K_t = K_t[:3, :3]
    w2c_t = torch.linalg.inv(c2w_t)
    N = pcd_world.shape[0]
    ones = torch.ones((N,1), device=device, dtype=dtype)
    Pw_h = torch.cat([pcd_world, ones], dim=1).t()
    Pc = (w2c_t @ Pw_h)[:3].t()  # (N,3)
    save_pointcloud_as_ply(Pc, "gt_scene_pcl_wrt_cam.ply")
    if z_forward == "+Z" or z_forward == "Z":
        depth_vals = Pc[:,2]; in_front = depth_vals > 0.05
    else:  # "-Z" tipico Habitat/OpenGL
        depth_vals = -Pc[:,2]; in_front = depth_vals > 0.05
        Pc = Pc.clone(); Pc[:,2] *= -1

    if in_front.sum().item() == 0:
        return torch.full((H,W), float('inf'), device=device, dtype=dtype), 0

    Pc = Pc[in_front]; dval = depth_vals[in_front]
    uv_h = (K_t @ Pc.t()).t()
    u = torch.round(uv_h[:,0] / uv_h[:,2]).long()
    v = torch.round(uv_h[:,1] / uv_h[:,2]).long()
    valid = (u>=0)&(u<W)&(v>=0)&(v<H)
    if valid.sum().item() == 0:
        return torch.full((H,W), float('inf'), device=device, dtype=dtype), 0

    u = u[valid]; v = v[valid]; dval = dval[valid]
    lin = v * W + u
    sort = torch.argsort(lin)
    lin = lin[sort]; dval = dval[sort]
    first = torch.ones_like(lin, dtype=torch.bool, device=device); first[1:] = lin[1:] != lin[:-1]
    z_exp = torch.full((H*W,), float('inf'), device=device, dtype=dtype); z_exp[lin[first]] = dval[first]
    return z_exp.view(H, W), int(torch.isfinite(z_exp).sum().item())

def render_env_depth_by_raycast(mesh: trimesh.Trimesh,
                                c2w_4x4: np.ndarray,
                                K_3x3: np.ndarray,
                                img_hw,
                                z_forward: str = "-Z",
                                stride: int = 2):
    H, W = int(img_hw[0]), int(img_hw[1])
    # intersector (pyembree se c'è, altrimenti fallback)
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(mesh)
    except Exception:
        from trimesh.ray.ray_triangle import RayMeshIntersector
        rmi = RayMeshIntersector(mesh)

    uu, vv = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    uv = np.stack([uu.ravel(), vv.ravel(), np.ones_like(uu).ravel()], axis=0).astype(np.float64)  # (3,N)

    Kinv = np.linalg.inv(K_3x3.astype(np.float64))
    dirs_cam = (Kinv @ uv).T
    dirs_cam /= (np.linalg.norm(dirs_cam, axis=1, keepdims=True) + 1e-12)

    if z_forward in ("-Z", "-z"):  # Habitat-style
        dirs_cam = dirs_cam.copy()
        dirs_cam[:, 2] *= -1.0

    T = np.asarray(c2w_4x4, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    ray_origins    = np.repeat(t[None, :], dirs_cam.shape[0], axis=0)
    ray_directions = (R @ dirs_cam.T).T  # (N,3)

    # distanza euclidea al primo impatto
    try:
        t_hit = rmi.intersects_first(ray_origins, ray_directions)  # (N,) float, -1 se miss
    except Exception:
        ids = rmi.intersects_id(ray_origins, ray_directions, multiple_hits=False)
        t_hit = np.full((ray_origins.shape[0],), -1.0, dtype=np.float64)
        if ids is not None and len(ids) > 0:
            idx = ids[0]
            loc = rmi.intersects_location(ray_origins[idx], ray_directions[idx])[0]
            d = np.linalg.norm(loc - ray_origins[idx], axis=1)
            t_hit[idx] = d

    # converti distanza lungo raggio in profondità z (asse camera)
    cos_z = dirs_cam[:, 2].clip(min=1e-12)  # già flippato se -Z
    z_samples = np.where(t_hit > 0, t_hit * cos_z, np.inf)

    z_env = np.full((H, W), np.inf, dtype=np.float64)
    z_env[vv.ravel(), uu.ravel()] = z_samples
    if stride > 1:
        # opzionale: upsample nearest-neighbor
        import cv2
        z_env = cv2.resize(z_env, (W, H), interpolation=cv2.INTER_NEAREST)
    return z_env

def check_camera_pose_wrt_map_with_mesh(env_mesh,        # trimesh.Trimesh già nel world giusto
                                        c2w_t,           # torch (4x4)
                                        inv_K_t,         # torch (3x3 o 4x4)
                                        depth_np,        # np (H,W)
                                        img_hw,
                                        z_forward="-Z",
                                        tau_m=0.03,
                                        stride=2,
                                        frame_id=0):
    import torch, numpy as np, cv2, os

    device = c2w_t.device; dtype = c2w_t.dtype
    H, W = img_hw
    save_mask_path = os.path.join("extracted_mask_from_mesh", f"novelty_mask_{frame_id}.png")
    # K 3x3 in numpy
    K = torch.linalg.inv(inv_K_t.to(device=device, dtype=dtype))[:3, :3].detach().cpu().numpy()
    # c2w in numpy
    c2w = c2w_t.detach().cpu().numpy()

    # depth attesa con occlusioni (mesh)
    z_env = render_env_depth_by_raycast(env_mesh, c2w, K, (H, W), z_forward=z_forward, stride=stride)  # np (H,W)

    # osservata
    z_obs = torch.from_numpy(depth_np).to(device=device, dtype=dtype)
    z_env_t = torch.from_numpy(z_env).to(device=device, dtype=dtype)

    obs_valid = torch.isfinite(z_obs) & (z_obs > 0)
    env_hit   = torch.isfinite(z_env_t)

    # NOVEL: ambiente prevede superficie a distanza z_env, ma vedo qualcosa più vicino
    cond = obs_valid & env_hit & ((z_obs + tau_m) < z_env_t)
    novelty_mask = (cond.detach().cpu().numpy().astype(np.uint8)) * 255

    if save_mask_path is not None:
        os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)
        cv2.imwrite(save_mask_path, novelty_mask)
        print("[novel] salvata:", save_mask_path)

    # metriche (dove c'è overlap valido)
    overlap = (env_hit.detach().cpu().numpy()) & (obs_valid.detach().cpu().numpy())
    if overlap.any():
        delta = (z_env_t - z_obs)[overlap]
        mae  = float(torch.mean(torch.abs(delta)).item())
        rmse = float(torch.sqrt(torch.mean(delta**2)).item())
    else:
        mae = rmse = float("nan")

    return {"overlap_px": int(overlap.sum()), "mae": mae, "rmse": rmse}, novelty_mask

def check_camera_pose_wrt_map(gt_3d_oriented_w: torch.Tensor,
                              c2w_t: torch.Tensor,
                              inv_K_t: torch.Tensor,
                              depth_np: np.ndarray,
                              img_hw,
                              z_forward="-Z",
                              tau_m=0.03, save_ply=True, frame_id=0):
    
    device = c2w_t.device; dtype = c2w_t.dtype
    K_t = torch.linalg.inv(inv_K_t.to(device=device, dtype=dtype))
    z_exp, valid_px = render_expected_depth_from_pcd(gt_3d_oriented_w.to(device=device, dtype=dtype), c2w_t, K_t, img_hw, z_forward=z_forward)
    H, W = img_hw
    if save_ply:
        save_expected_and_observed_depth_as_ply(
            z_exp=z_exp,
            depth_np=depth_np,
            inv_K_t=inv_K_t,
            c2w_t=c2w_t,
            img_hw=(H, W),
            out_dir=os.path.join("ply_depths"),
            z_forward=z_forward,
            stride=2, frame_id=frame_id
        )

    z_obs = torch.from_numpy(depth_np).to(device=device, dtype=dtype)
    valid = torch.isfinite(z_exp) & torch.isfinite(z_obs)
    if valid.sum().item() == 0:
        print("[pose-check] nessuna sovrapposizione valida.")
        return {"valid_px": 0}
    delta = (z_exp - z_obs)[valid]
    mae = torch.mean(torch.abs(delta)).item()
    rmse = torch.sqrt(torch.mean(delta**2)).item()
    inlier = torch.mean((torch.abs(delta) < tau_m).float()).item()
    print(f"[pose-check] valid_px={valid_px} | MAE={mae:.3f} m | RMSE={rmse:.3f} m | Inliers(|Δ|<{tau_m*100:.0f}cm)={inlier*100:.1f}%")

    novelty = torch.zeros_like(z_exp, dtype=torch.bool)
    cond = valid & ((z_obs+0.5) < z_exp)
    novelty[cond] = True
    novelty_mask = novelty.detach().cpu().numpy().astype(np.uint8) * 255  # (H,W) 0/255
    save_mask_path = os.path.join("extracted_mask", f"novelty_mask_{frame_id}.png")
    os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)
    cv2.imwrite(save_mask_path, novelty_mask)
    print(f"[pose-check] salvata novelty mask: {save_mask_path}")

    return {"valid_px": int(valid_px), "mae": mae, "rmse": rmse, "inlier": inlier}

@torch.no_grad()
def novelty_mask_from_pcd_nn(env_pcd_xyz: torch.Tensor,     # torch (N,3) WORLD
                             depth_np: np.ndarray,          # (H,W) float32 [m]
                             inv_K_t: torch.Tensor,         # torch (3x3 o 4x4)
                             c2w_t: torch.Tensor,           # torch (4x4)
                             img_hw,                        # (H, W)
                             z_forward: str = "-Z",
                             dist_thresh_m: float = 0.05,   # 5 cm
                             stride: int = 1,               # sottocampionamento per velocità
                             frame_id: int = 0):
    """
    Genera una mask BN (H,W) dove 255 = punto osservato non spiegato dalla PCD ambiente entro dist_thresh_m.
    Usa NN (cKDTree) sull'ambiente: veloce e senza raycasting.
    """
    import cv2, os

    H, W = int(img_hw[0]), int(img_hw[1])

    # --- back-project osservato -> punti WORLD (sottocampionati) ---
    device = c2w_t.device
    dtype  = c2w_t.dtype
    z_obs  = torch.from_numpy(depth_np).to(device=device, dtype=dtype)

    uu, vv = torch.meshgrid(
        torch.arange(0, W, stride, device=device),
        torch.arange(0, H, stride, device=device),
        indexing="xy"
    )
    d = z_obs[vv, uu]
    valid = torch.isfinite(d) & (d > 0)
    save_mask_path = os.path.join("extracted_mask", f"novelty_mask_{frame_id}.png")
    if valid.sum().item() == 0:
        mask = np.zeros((H, W), dtype=np.uint8)
        return mask

    u = uu[valid].float()
    v = vv[valid].float()
    d = d[valid]

    K_t = torch.linalg.inv(inv_K_t.to(device=device, dtype=dtype))[:3, :3]
    Kinv = torch.linalg.inv(K_t)

    rays = (Kinv @ torch.stack([u, v, torch.ones_like(u)], dim=0)).t()  # (M,3)
    Pc   = rays * d.unsqueeze(1)
    if z_forward in ("-Z", "-z"):
        Pc = Pc.clone(); Pc[:, 2] *= -1

    ones = torch.ones((Pc.shape[0], 1), device=device, dtype=dtype)
    Pw   = (c2w_t @ torch.cat([Pc, ones], dim=1).t()).t()[:, :3]        # (M,3) WORLD

    # --- KDTree su env PCD (CPU) ---
    P_env = env_pcd_xyz.detach().cpu().numpy()
    tree  = cKDTree(P_env)

    P_obs = Pw.detach().cpu().numpy()
    dists, _ = tree.query(P_obs, k=1, workers=-1)  # veloce

    novel_flags = dists > dist_thresh_m  # (M,)

    # --- ricostruisci mask BN su (H,W) ---
    Hs = (H + stride - 1) // stride
    Ws = (W + stride - 1) // stride
    mask_sub = np.zeros((Hs, Ws), dtype=np.uint8)

    # indice booleano 2D (subsampled) dei pixel validi
    valid_sub = valid.detach().cpu().numpy()                  # (Hs, Ws)
    mask_sub[valid_sub] = novel_flags                         # assegna solo nei True

    # upsample alla risoluzione piena
    # mask = cv2.resize(mask_sub * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = mask_sub
    # Check enough novel pixels
    min_novel_px = 20
    if (mask > 0).sum() < min_novel_px:
        return np.zeros((H, W), dtype=np.uint8)

    return mask


@torch.no_grad()
def save_expected_and_observed_depth_as_ply(z_exp: torch.Tensor,
                                            depth_np,                 # numpy (H,W)
                                            inv_K_t: torch.Tensor,    # (3,3) o (4,4) torch
                                            c2w_t: torch.Tensor,      # (4,4) torch
                                            img_hw,                   # (H, W)
                                            out_dir: str,
                                            z_forward: str = "-Z",
                                            stride: int = 2,
                                            frame_id: int = 0):
    """
    Salva due PLY (expected/observed) back-proiettando z_exp e depth osservata.
    - z_forward: usa "-Z" per convenzione OpenGL/Habitat, "+Z" per pinhole classico.
    - stride: sottocampionamento pixel (2=prende 1 px ogni 2 per asse).
    Output: out_dir/expected.ply, out_dir/observed.ply
    """
    import numpy as np
    import open3d as o3d

    device = c2w_t.device
    dtype  = c2w_t.dtype
    H, W   = int(img_hw[0]), int(img_hw[1])

    # Tensors coerenti su device
    z_exp = z_exp.to(device=device, dtype=dtype)
    z_obs = torch.from_numpy(depth_np).to(device=device, dtype=dtype)

    K_t = torch.linalg.inv(inv_K_t.to(device=device, dtype=dtype))
    K_t = K_t[:3, :3]
    Kinv = torch.linalg.inv(K_t)

    # griglia sottocampionata
    uu, vv = torch.meshgrid(
        torch.arange(0, W, stride, device=device),
        torch.arange(0, H, stride, device=device),
        indexing="xy"
    )

    def depth_to_world_points(z_map: torch.Tensor):
        d = z_map[vv, uu]
        valid = torch.isfinite(d) & (d > 0)
        if valid.sum().item() == 0:
            return torch.empty((0,3), device=device, dtype=dtype)
        u = uu[valid].float()
        v = vv[valid].float()
        d = d[valid]
        rays = (Kinv @ torch.stack([u, v, torch.ones_like(u)], dim=0)).t()  # (N,3)
        Pc = rays * d.unsqueeze(1)
        if z_forward == "-Z":
            Pc = Pc.clone(); Pc[:,2] *= -1
        Pc_h = torch.cat([Pc, torch.ones((Pc.shape[0],1), device=device, dtype=dtype)], dim=1).t()
        Pw = (c2w_t @ Pc_h).t()[:, :3]
        return Pw

    Pw_exp = depth_to_world_points(z_exp)
    Pw_obs = depth_to_world_points(z_obs)

    os.makedirs(out_dir, exist_ok=True)
    if Pw_exp.shape[0] > 0:
        pcd_e = o3d.geometry.PointCloud()
        pcd_e.points = o3d.utility.Vector3dVector(Pw_exp.detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(out_dir, f"expected_{frame_id}.ply"), pcd_e)
    else:
        print("[PLY] expected: nessun punto valido.")

    if Pw_obs.shape[0] > 0:
        pcd_o = o3d.geometry.PointCloud()
        pcd_o.points = o3d.utility.Vector3dVector(Pw_obs.detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(out_dir, f"observed_{frame_id}.ply"), pcd_o)
    else:
        print("[PLY] observed: nessun punto valido.")

    print(f"[PLY] salvati in: {out_dir}/expected.ply e {out_dir}/observed.ply")