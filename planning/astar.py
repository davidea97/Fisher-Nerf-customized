import json
import tempfile
import numpy as np
import torch
import heapq
import cv2
import os
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, List
from models.SLAM.utils.slam_external import build_rotation
import time
import datasets.util.map_utils as map_utils
import math

from .planning_utils import color_mapping_3, heatmap, LocalizationError, combimed_heuristic
from .max_min_dist import select_maximin_points_vectorized, min_dist_center_approximate
        
from frontier_exploration.frontier_search import FrontierSearch

class AstarPlanner:
    def __init__(self, 
                 slam_config,
                 eval_dir,
                 device=torch.device("cuda:0")) -> None:
        """
        A star planning on occ_map
            start in [y, x] order
            occ_map -- 1 - occupied; 0 - free
        """
        
        self.device = device
        self.cell_size = slam_config["explore"]["cell_size"]
        self.height_upper = slam_config["policy"]["height_upper"]
        self.height_lower = slam_config["policy"]["height_lower"]
        self.add_random_gaussians = slam_config["explore"]["add_random_gaussians"]

        self.K = slam_config["explore"]["sample_view_num"]
        self.K_object = slam_config["explore_object"]["sample_view_num"]
        self.radius = slam_config["explore"]["sample_range"]
        self.radius_object = slam_config["explore_object"]["sample_range"]
        self.eval_dir = eval_dir
        self.min_range = slam_config["explore"]["min_range"]
        self.min_range_object = slam_config["explore_object"]["min_range"]
        self.occ_map_np = None

        self.centering = slam_config["explore"]["centering"]
        self.frontier_select_method = slam_config["explore"]["frontier_select_method"]

        self.cam_pos = None # camera coordinate on the map [x, z] np.int8
        self.shortcut_path = slam_config["explore"]["shortcut_path"]
        self.pcd_far_distance = slam_config["policy"]["pcd_far_distance"]

        self.previous_candidates = None
        if self.frontier_select_method == "vlm":
            self.vlm = VLMFrontierSelection()

        # DAVIDE
        self.covered=None
        self.known_env = False
        self.frontier_radius = 1


    def init(self, pose, intrinsic, scene_bounds = None):
        """ 
        Init the Astar Planner 
        
        Args:
            pose: (4, 4) torch.Tensor, the camera pose in world coordinate
            intrinsic: (3, 3) torch.Tensor, the camera intrinsic matrix
            scene_bounds: (2, 3) np.ndarray, the scene bounds
        """
        # set up bounds for scene
        self.grid_dim = np.array([768, 768])
        self.intrinsics = intrinsic
        self.cam_height = pose[1, 3]

        # if 
        if scene_bounds is not None:
            self.scene_bounds = scene_bounds
            scene_lower, scene_upper = scene_bounds
            map_center_np = (scene_upper[[0, 2]] + scene_lower[[0, 2]]) / 2 # x-z coordinate for the center of the map

            # here, we choose to use fixed resolution
            grid_x = (scene_upper[0] - scene_lower[0]) / self.cell_size
            grid_z = (scene_upper[2] - scene_lower[2]) / self.cell_size
            self.grid_dim = np.array([int(grid_x+1), int(grid_z+1)])
        else:
            # The initial position is at the center
            map_center_np = pose[[0, 2], 3]
        
        # initialize map
        self.occ_map = torch.zeros((3, self.grid_dim[1], self.grid_dim[0]), device=self.device)
        # all map cells are initialized as unknown
        self.occ_map[0] = 1.

        cam_pos_x = int((pose[0, 3] - map_center_np[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((pose[2, 3] - map_center_np[1]) / self.cell_size + self.grid_dim[1] // 2)
        self.cam_pos = np.array([cam_pos_z, cam_pos_x])

        # set the current robot location as free
        self.occ_map[2, cam_pos_z-1:cam_pos_z+2, cam_pos_x-1:cam_pos_x+2] = 2.

        self.map_center = torch.from_numpy(map_center_np).to(self.device)
        self.frame_idx = 0

    def init_known_env_from_known_env(self, pose, env_pcd_world, max_lines=20000):
        """ 
        Init the Astar Planner 
        
        Args:
            pose: (4, 4) torch.Tensor, the camera pose in world coordinate
            intrinsic: (3, 3) torch.Tensor, the camera intrinsic matrix
            scene_bounds: (2, 3) np.ndarray, the scene bounds
        """
        # set up bounds for scene
        self.grid_dim = np.array([768, 768])
        H, W = self.grid_dim
        grid   = torch.empty(3, H, W, device=self.device)
        # The initial position is at the center
        map_center_np = pose[[0, 2], 3]
        
        # initialize map
        self.occ_map = torch.zeros((3, self.grid_dim[1], self.grid_dim[0]), device=self.device)
        occ_map = torch.zeros_like(self.occ_map)
        # all map cells are initialized as unknown
        self.occ_map[0] = 1.

        cam_pos_x = int((pose[0, 3] - map_center_np[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((pose[2, 3] - map_center_np[1]) / self.cell_size + self.grid_dim[1] // 2)
        self.cam_pos = np.array([cam_pos_z, cam_pos_x])

        # set the current robot location as free
        self.occ_map[2, cam_pos_z-1:cam_pos_z+2, cam_pos_x-1:cam_pos_x+2] = 2.

        self.map_center = torch.from_numpy(map_center_np).to(self.device)
        self.frame_idx = 0

        grid.fill_(0.)
        pc = env_pcd_world.to(self.device, dtype=torch.float32)
        map_coords = map_utils.discretize_coords(
            pc[:, 0], pc[:, 2], self.grid_dim, self.cell_size, self.map_center
        )  # (M,2) int: [ix, iz]

        # all particles are treated as free
        occ_lbl = torch.ones((pc.shape[1], 1), device=pc.device).long() * 2
        y = pc[:, 1]
        valid_sgn = torch.bitwise_and(y >= self.height_lower, y <= self.height_upper)
        pc_occ = pc[valid_sgn]  # (M,3)

        concatenated = torch.cat([map_coords[valid_sgn], occ_lbl[valid_sgn]], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5

        # accumula sul temporaneo
        occ_map += 0.01 * grid
        
        grid.fill_(0.)
        

        # ---- FREE: linee dal robot alle celle occupied (come fai tu) ----
        # prendi le (z,x) delle celle occupied uniche
        occ_z = unique_values[:, 1].detach().cpu().numpy()
        occ_x = unique_values[:, 0].detach().cpu().numpy()

        # opzionale: limita numero di linee per performance
        if max_lines is not None and len(occ_z) > max_lines:
            idx = np.random.choice(len(occ_z), size=max_lines, replace=False)
            occ_z = occ_z[idx]; occ_x = occ_x[idx]

        line_canvas = np.zeros((H, W), dtype=np.uint8)
        for z, x in zip(occ_z, occ_x):
            p_z, p_x = int(z), int(x)
            # disegna la linea 2D come nel tuo update
            line_canvas = cv2.line(line_canvas, (p_x, p_z), (cam_pos_x, cam_pos_z), 1, 1)

        free_z, free_x = np.where(line_canvas > 0)
        occ_map[2, free_z, free_x] = 1.  # free lungo i raggi

        # ---- normalizzazione e fusione come nel tuo codice ----
        denom = (occ_map.sum(dim=0, keepdim=True) + 1e-5)
        self.occ_map += occ_map / denom

    def save(self, save_path):
        torch.save(
            {
                "occ_map": self.occ_map,
                "map_center": self.map_center,
                "frame_idx": self.frame_idx,
            },
            os.path.join(save_path, "astar.pth")
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.occ_map = checkpoint["occ_map"]
        self.map_center = checkpoint["map_center"]
        self.frame_idx = checkpoint["frame_idx"]

    @torch.no_grad()
    def update_occ_map(self, depth, c2w, t, downsample=1):
        """ Update Occulision map based on depth observation """
        if c2w.device != self.device:
            c2w = c2w.to(self.device)
        
        self.frame_idx = t

        # update current robot location on occ_map
        cam_x, cam_z = c2w[0, 3], c2w[2, 3]
        cam_pos_x = int((cam_x - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((cam_z - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2)
        self.cam_pos = np.array([cam_pos_z, cam_pos_x])
        self.occ_map[2, cam_pos_z-1:cam_pos_z+2, cam_pos_x-1:cam_pos_x+2] = 1e3 # It is setting an higher value on the free map in the agent position
        
        # convert depth to torch Tensor
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        depth = depth.to(self.device)

        # # generate point cloud in current frame using points
        width, height = depth.shape[2], depth.shape[1]
        CX = self.intrinsics[0][2]
        CY = self.intrinsics[1][2]
        FX = self.intrinsics[0][0]
        FY = self.intrinsics[1][1]

        # Generate pts from depth image
        # Compute indices of pixels
        x_grid, y_grid = torch.meshgrid(torch.arange(0, width, step=downsample, device=self.device).float(), 
                                        torch.arange(0, height, step=downsample, device=self.device).float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY

        # sample z along depth
        sampled_z = torch.linspace(1e-3, 0.95, 11).reshape(-1, 1, 1).cuda() # (K, H, W)
        sampled_z.clamp_(min=0.)
        sampled_z[-1, 0, 0] = 1.              # add a point at the end of the depth range           

        xx, yy = xx.unsqueeze(0), yy.unsqueeze(0)  # (1, H, W)
        depth_z = sampled_z * depth[:, ::downsample, ::downsample]  # (K, H, W)
        mask = torch.bitwise_and(depth_z > 0, depth_z < self.pcd_far_distance)  # (K, H, W)

        pts = torch.stack((xx * depth_z, yy * depth_z, depth_z, torch.ones_like(depth_z)), dim = 0) # 4 x K x H x W
        free_particles = pts[:, :-1, :, :].reshape(4, -1)
        depth_pts = pts[:, -1, :, :].reshape(4, -1)

        # perform masking
        free_particles = free_particles[:, mask[:-1].reshape(-1)]
        depth_pts = depth_pts[:, mask[-1].reshape(-1)]

        grid = torch.empty(3, self.grid_dim[1], self.grid_dim[0], device=depth_pts.device)  
        occ_map = torch.zeros_like(self.occ_map) 

        # free particles updating
        grid.fill_(0.)  
        free_particles = c2w @ free_particles
        map_coords = map_utils.discretize_coords(free_particles[0], free_particles[2], self.grid_dim, self.cell_size, self.map_center)

        # all particles are treated as free
        occ_lbl = torch.ones((free_particles.shape[1], 1), device=free_particles.device).long() * 2
        # Remove floor and ceiling
        valid_sgn = torch.bitwise_and(free_particles[1] >= self.height_lower, free_particles[1] <= self.height_upper)

        # (N, 3) - (x, y, label)
        concatenated = torch.cat([map_coords[valid_sgn], occ_lbl[valid_sgn]], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5

        # update top-down map
        occ_map += 0.01 * grid

        # Update the top down map
        grid.fill_(0.)

        depth_pts = c2w @ depth_pts
        # Assign value 1 = OCCUPIED to depth points
        occ_lbl = torch.ones((depth_pts.shape[1], 1), device=depth_pts.device).long()
        valid_sgn = torch.bitwise_and(depth_pts[1] >= self.height_lower, depth_pts[1] <= self.height_upper)

        map_coords = map_utils.discretize_coords(depth_pts[0], depth_pts[2], self.grid_dim, self.cell_size, self.map_center)
        concatenated = torch.cat([map_coords[valid_sgn], occ_lbl[valid_sgn]], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5
        grid[1] *= 100

        # update top-down map
        occ_map += grid

        # draw line for the occ_map
        # maybe parallel here
        occ_z, occ_x = unique_values[:, 1].cpu(), unique_values[:, 0].cpu()
        line_canvas = np.zeros((self.grid_dim[1], self.grid_dim[0]), dtype=np.uint8)
        for z, x in zip(occ_z, occ_x):
            p_z, p_x = z.item(), x.item()
            line_canvas = cv2.line(line_canvas, (p_x, p_z), (cam_pos_x, cam_pos_z), 1, 1)
        free_z, free_x = np.where(line_canvas > 0)
        occ_map[2, free_z, free_x] = 1.

        self.occ_map += occ_map / (occ_map.sum(dim=0, keepdim=True) + 1e-5)

    ######## DAVIDE ########
    def _grid_ij_from_world(self, x, z):
        gx = int((x - self.map_center[0].item()) / self.cell_size + self.grid_dim[0] // 2)
        gz = int((z - self.map_center[1].item()) / self.cell_size + self.grid_dim[1] // 2)
        return gx, gz

    def _yaw_from_pose(self, c2w):
        # forward -Z nel frame camera → world
        # usa get_agent_state().rotation se preferisci; qui dedotto da c2w
        R = c2w[:3, :3].detach().cpu().numpy()
        fwd = R @ np.array([0, 0, -1], dtype=np.float32)
        return float(np.arctan2(fwd[2], fwd[0]))  # atan2(z,x)

    def _bresenham(self, i0, j0, i1, j1):
        di, dj = abs(i1-i0), abs(j1-j0)
        si, sj = (1 if i1>=i0 else -1), (1 if j1>=j0 else -1)
        err = di - dj
        i, j = i0, j0
        while True:
            yield i, j
            if i==i1 and j==j1: break
            e2 = 2*err
            if e2 > -dj: err -= dj; i += si
            if e2 <  di: err += di; j += sj
    
    def cover_fov_2d(self, c2w: torch.Tensor, fov_deg=90.0, max_range=4.0, ang_step_deg=2.0):
        H, W = self.covered.shape
        x, z = c2w[0,3].item(), c2w[2,3].item()
        gx, gz = self._grid_ij_from_world(x, z)
        if not (0 <= gx < W and 0 <= gz < H): return
        yaw = self._yaw_from_pose(c2w)
        half = np.deg2rad(fov_deg)*0.5
        angs = np.arange(-half, half+1e-6, np.deg2rad(ang_step_deg), dtype=np.float32)
        for da in angs:
            a = yaw + da
            x1 = x + max_range*np.cos(a); z1 = z + max_range*np.sin(a)
            g1x, g1z = self._grid_ij_from_world(x1, z1)
            for i,j in self._bresenham(gx, gz, g1x, g1z):
                if not (0<=i<W and 0<=j<H): break
                if self.occ_map[2, j, i] > 0:  # free
                    self.covered[j, i] = True
                else:
                    break
    
    def build_frontier_cells(self):
        """Ritorna lista di (j,i) fronteira: free & !covered & adiacenti a covered."""
        H, W = self.covered.shape
        covered = self.covered
        free = self.occ_map[2] > 0
        unknown = ~covered  # inesplorato = non visto
        # adiacenza 4-neigh
        adj = torch.zeros_like(covered)
        adj[:-1] |= covered[1:]
        adj[1:]  |= covered[:-1]
        adj[:, :-1] |= covered[:, 1:]
        adj[:, 1:]  |= covered[:, :-1]
        fr = (unknown & free & adj)
        js, is_ = torch.where(fr)
        return list(zip(js.tolist(), is_.tolist()))
    ##########################

    def build_connected_occupied_space(self, gaussian_points=None):
        
        prob, index = self.occ_map.max(dim=0)
        
        index = index.cpu().numpy()
        free_space = (index == 2)
        occupied_space = (index == 1)

        # project 3D gaussians to build frontiers
        # height_range = self.config["explore"]["height_range"]
        if gaussian_points is not None:
            # lower_y, upper_y = self.cam_height - 1.0, self.cam_height
            selected_points = gaussian_points
            map_coords = map_utils.discretize_coords(selected_points[:, 0], selected_points[:, 2], self.grid_dim, self.cell_size, self.map_center)
            unique_values, counts = torch.unique(map_coords, dim=0, return_counts=True)
            # add 25 filter; since tha gaussians points might block narrow hallway.

            unique_values = unique_values.cpu().numpy()
            occupied_space[unique_values[:, 1], unique_values[:, 0]] = 1

        # perform Open morph on free space
        kernel = np.ones((3, 3), np.uint8)
        occupied_space = cv2.morphologyEx(occupied_space.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # get the connected region of the current robot
        _, labels, stats, centroid = cv2.connectedComponentsWithStats(occupied_space.astype(np.uint8))
        
        # select the one with largest size
        label_index = np.argsort(stats[:, 4])

        # largest forground label
        robot_label = label_index[-1] if label_index[-1] != 0 else label_index[-2]
        occupied_space = (labels == robot_label).astype(np.uint8)

        # free_space = cv2.erode(free_space.astype(np.uint8), np.ones((3, 3), np.uint8))
        return occupied_space
    
    def build_connected_freespace(self, gaussian_points=None):
        """ find the connected free space to the robot 
        
        Args:
            gaussian_points: (N, 3) torch.Tensor, the 3D gaussian points
        
        Returns:
            free_space: (H, W) np.ndarray, the connected free space 
                1 - free
                0 - occupied
        """
        prob, index = self.occ_map.max(dim=0)
        
        index = index.cpu().numpy()
        free_space = (index == 2)
        unkown = (index == 0)

        # project 3D gaussians to build frontiers
        # height_range = self.config["explore"]["height_range"]
        if free_space.sum() > 18 and gaussian_points is not None:
            # lower_y, upper_y = self.cam_height - 1.0, self.cam_height
            sign = torch.bitwise_and(gaussian_points[:, 1] >= self.height_lower, gaussian_points[:, 1] <= self.height_upper)
            selected_points = gaussian_points[sign]
            map_coords = map_utils.discretize_coords(selected_points[:, 0], selected_points[:, 2], self.grid_dim, self.cell_size, self.map_center)
            unique_values, counts = torch.unique(map_coords, dim=0, return_counts=True)
            # add 25 filter; since tha gaussians points might block narrow hallway.
            unique_values = unique_values[counts > 25]

            unique_values = unique_values.cpu().numpy()
            free_space[unique_values[:, 1], unique_values[:, 0]] = 0

        # perform Open morph on free space
        kernel = np.ones((3, 3), np.uint8)
        free_space = cv2.morphologyEx(free_space.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # get the connected region of the current robot
        _, labels, stats, centroid = cv2.connectedComponentsWithStats(free_space.astype(np.uint8))
        
        # select the one with largest size
        label_index = np.argsort(stats[:, 4])

        # largest forground label
        robot_label = label_index[-1] if label_index[-1] != 0 else label_index[-2]
        free_space = (labels == robot_label).astype(np.uint8)

        # free_space = cv2.erode(free_space.astype(np.uint8), np.ones((3, 3), np.uint8))
        return free_space

    def setup_start(self, start, gaussian_points = None, frame_idx = 0):  
        """ Setup the start point for the planner 
            This function should be called before the planning
        
        """ 
        occ_map = self.occ_map.argmax(0) == 1
        
        self.occ_map_np = occ_map.cpu().numpy()
        H, W = self.occ_map_np.shape
        self.start = start

        self.planning_direction = np.ones((H, W, 4)) * -1 # (cost, parent_y, parent_x, collision_risk)
        self.planning_direction[self.start[0], self.start[1]] = [0, self.start[0], self.start[1], 0]

        occ_map = (self.occ_map.argmax(dim=0) == 1)

        if gaussian_points is not None:
            # project 3D gaussians onto the 2D Map
            # cast 3D Gaussians to ground
            lower_y, upper_y = self.cam_height - 1.0, self.cam_height
            sign = torch.bitwise_and(gaussian_points[:, 1] >= lower_y, gaussian_points[:, 1] <= upper_y)
            selected_points = gaussian_points[sign]
            map_coords = map_utils.discretize_coords(selected_points[:, 0], selected_points[:, 2], self.grid_dim, self.cell_size, self.map_center)
            unique_values, counts = torch.unique(map_coords, dim=0, return_counts=True)
            occ_map[unique_values[counts > 50, 1], unique_values[counts > 50, 0]] = 1

        binarymap = occ_map.cpu().numpy().astype(np.uint8)
        
        # dilate binary map
        binarymap = cv2.dilate(binarymap, np.ones((3, 3), np.uint8))
        local_patch = binarymap[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
        local_patch[1, 1] = 0
        if local_patch.sum() >= 8:
            raise LocalizationError("The start point is not in free space")
        binarymap[start[0], start[1]] = 0

        # set occ map np
        self.occ_map_np = binarymap

        # ========= VISUALIZZAZIONE GAUSSIANI SULLA OCC MAP =========
        try:
            os.makedirs(os.path.join(self.eval_dir, "occ_map"), exist_ok=True)

            # base: occ_map PRIMA della dilatazione, utile per capire dove scriviamo le celle
            base_occ = (self.occ_map.argmax(dim=0) == 1).cpu().numpy().astype(np.uint8)

            # colore: da binario -> RGB (bianco = occupato, nero = libero)
            vis = np.zeros((base_occ.shape[0], base_occ.shape[1], 3), dtype=np.uint8)
            vis[base_occ != 0] = (255, 255, 255)

            # se abbiamo gaussian_points, proiettiamoli per plotting
            if gaussian_points is not None and sign.any():
                # coords di TUTTI i gaussiani selezionati (altezza nel range)
                all_uv = map_coords.detach().cpu().numpy()  # (N,2) [x_idx, y_idx]
                # clip a bordo
                all_uv[:, 0] = np.clip(all_uv[:, 0], 0, self.grid_dim[0]-1)
                all_uv[:, 1] = np.clip(all_uv[:, 1], 0, self.grid_dim[1]-1)

                # coords che superano la soglia counts>50 (quelle che hai usato per settare occupato)
                unique_uv = unique_values.detach().cpu().numpy()
                unique_uv[:, 0] = np.clip(unique_uv[:, 0], 0, self.grid_dim[0]-1)
                unique_uv[:, 1] = np.clip(unique_uv[:, 1], 0, self.grid_dim[1]-1)

                # Disegna i gaussiani proiettati (tutti) in CIANO
                for x, y in all_uv:
                    cv2.circle(vis, (int(x), int(y)), 1, (255, 255, 0), -1)  # BGR: (0,255,255) ma invertito -> (255,255,0)

                # Disegna quelli sopra soglia (magenta) per evidenziare cosa hai marcato occupato
                # Nota: qui usiamo counts>50 come in occ_map[...] = 1
                keep = (counts > 50).detach().cpu().numpy()
                uv_th = unique_uv[keep]
                for x, y in uv_th:
                    cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 255), -1)  # magenta

            # start (ricorda: start è in [y, x])
            cv2.drawMarker(vis, (int(self.start[1]), int(self.start[0])), (255, 0, 0),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)  # blu

            # anche la mappa dopo la dilatazione come reference
            vis_dil = np.zeros((binarymap.shape[0], binarymap.shape[1], 3), dtype=np.uint8)
            vis_dil[binarymap != 0] = (255, 255, 255)

            # salva
            import matplotlib.pyplot as plt
            plt.imsave(os.path.join(self.eval_dir, "occ_map", f"occmap_gaussians_raw_{frame_idx}.png"), vis[..., ::-1])     # BGR->RGB
            plt.imsave(os.path.join(self.eval_dir, "occ_map", f"occmap_after_dilate_{frame_idx}.png"), vis_dil)
        except Exception as e:
            print("[WARN] Plot gaussians on occ map failed:", e)

        self.free_space_np = self.build_connected_freespace(gaussian_points)

    def build_frontiers(self, gaussian_points = None):
        """ Return frontiers in pixel space  """
        # find the connected free space
        free_space = self.build_connected_freespace(gaussian_points)
        
        # plt.figure()
        # plt.imshow(free_space)
        # plt.savefig(os.path.join(self.eval_dir, "freespace_{}.png".format(self.frame_idx)))
        # plt.close()

        # find the unknown area
        prob, index = self.occ_map.max(dim=0)
        index = index.cpu().numpy()
        unknown = (index == 0)

        map_center = self.map_center.cpu().numpy()
        # perform dilation
        kernel = np.ones((3, 3), np.uint8)
        free_space_dilate = cv2.dilate(free_space.astype(np.uint8), kernel, iterations=1)
        boundary = free_space_dilate - free_space
        frontier = np.bitwise_and(boundary, unknown)
        self.frontier = frontier

        # If no frontier found, return None
        if frontier.sum() == 0:
            self.target_frontier = None
            return None, free_space
        
        kernel = np.ones((3, 3), np.uint8)  
        frontier = cv2.dilate(frontier.astype(np.uint8), kernel, iterations=1) 
        
        # store frontier
        # cv2.imwrite(os.path.join(self.eval_dir, "frontier_{}.png".format(self.frame_idx)), frontier.astype(np.uint8) * 255)

        # find connected components in frontier
        num_labels, labels = cv2.connectedComponents(frontier.astype(np.uint8))
        unique_label, counts = np.unique(labels, return_counts=True)
        # remove the background
        unique_label = unique_label[1:]
        counts = counts[1:]

        # # area filter
        min_area = 10
        unique_label = unique_label[counts > min_area]
        if len(unique_label) == 0:
            return None, free_space
        counts = counts[counts > min_area]

        # Find the largest connected component
        if self.frontier_select_method == "largest":
            label_idx = np.argsort(counts)[::-1]
            self.selection = 0
            select_index = min(self.selection, len(label_idx) - 1)
            largest_label = unique_label[label_idx[select_index]]
            largest_frontier = (labels == largest_label).astype(np.uint8)

            # Record Frontiers for visualization
            self.target_frontier = largest_frontier
        
        elif self.frontier_select_method == "combined":
            max_score = 0
            target_label = -1
            
            for label, count in zip(unique_label, counts):
                frontier_pos = np.stack(np.where(labels == label), axis=1) # (K, 2)
                if len(frontier_pos) < 4:
                    continue
                
                frontier_distance = np.linalg.norm(frontier_pos - self.cam_pos, axis=1)
                mean_distance = frontier_distance.mean()
                # calculate area score size/distance
                area_score = (count) / (mean_distance + 20)

                if area_score > max_score:
                    max_score = area_score
                    target_label = label

            # No target selected, continue
            if target_label == -1:
                return None, free_space

            best_frontier = (labels == target_label).astype(np.uint8)
            # Record Frontiers for visualization
            self.target_frontier = best_frontier
        
        elif self.frontier_select_method == "closest":
            max_distance = 1e4
            target_label = -1

            for label in unique_label:
                frontier_pos = np.stack(np.where(labels == label), axis=1) # (K, 2)
                if len(frontier_pos) < 4:
                    continue
                
                frontier_distance = np.linalg.norm(frontier_pos - self.cam_pos, axis=1)
                mean_distance = frontier_distance.mean()

                if mean_distance < max_distance:
                    max_distance = mean_distance
                    target_label = label
            
            # No target selected, continue
            if target_label == -1:
                return None, free_space

            closest_frontier = (labels == target_label).astype(np.uint8)
            # Record Frontiers for visualization
            self.target_frontier = closest_frontier

        # find the center of the selected connected component
        select_pixels = np.stack(np.where(self.target_frontier), axis=1)
        # center = select_pixels.mean(axis=0)
        # center = center[[1, 0]] # switch to x,z

        select_pixels = select_pixels[:, [1, 0]]
        select_pixels = (select_pixels - np.array([[self.grid_dim[0] // 2, self.grid_dim[1] // 2]])) * self.cell_size + map_center[None, :]
        
        # FBE LOGIC
        if gaussian_points is None: 
            
            agent_pos = self.cam_pos  # expected in (x, z)
            min_thresh = 0.5  # meters, adjust as needed

            # Compute distances
            distances = np.linalg.norm(select_pixels - agent_pos[None, :], axis=1)

            # Find frontiers beyond threshold
            valid_idx = np.where(distances >= min_thresh)[0]

            if len(valid_idx) > 0:
                # Pick the closest valid frontier
                best_idx = valid_idx[np.argmin(distances[valid_idx])]
                frontier_point = select_pixels[best_idx:best_idx+1]  # shape (1,2)
            else:
                # Fallback: go in opposite direction
                print("No frontier beyond min_thresh. Going backward.")
                angle = math.pi * 5/4
                x, y = math.cos(angle), math.sin(angle)
                opposite_dir = np.array([[-x, -y]]) * 0.5  # random_magnitude
                frontier_point = agent_pos[None, :] + opposite_dir  # shape (1,2)
        else:
            frontier_point = select_pixels

        return frontier_point, free_space

    
    def build_object_frontiers(self, gaussian_points, use_convex_hull=True):

        # free space, per coerenza con build_frontiers (lo ritorniamo alla fine)
        # free_space = self.build_connected_freespace(gaussian_points)

        # guardie
        if gaussian_points is None or gaussian_points.numel() == 0:
            return None

        pts = gaussian_points

        # mondo -> pixel (u,v) = (x_idx, y_idx)
        map_coords = map_utils.discretize_coords(
            pts[:, 0],   # x
            pts[:, 2],   # z
            self.grid_dim,
            self.cell_size,
            self.map_center
        )  # (K,2)
        map_coords_unique, counts = torch.unique(map_coords, dim=0, return_counts=True)
        map_coords_unique = map_coords_unique[counts > 3]  # (N,2) [u,v] = [x_idx,y_idx]
        # mask oggetto (H,W) senza alcuna morfologia
        H, W = self.grid_dim[1], self.grid_dim[0]
        object_mask = np.zeros((H, W), dtype=np.uint8)

        uv = map_coords_unique.detach().cpu().numpy().astype(int)
        uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)
        object_mask[uv[:, 1], uv[:, 0]] = 1  # [y, x] = [v, u]
        

        # se non ci sono pixel accesi → None
        if object_mask.sum() == 0:
            return None

        # prendi TUTTI i pixel dell'oggetto (come per le frontiere)
        select_pixels = np.stack(np.where(object_mask), axis=1)  # (N,2) [y, x]
        # switch a [x, y]
        select_pixels = select_pixels[:, [1, 0]]

        # pixel -> mondo (stessa formula di build_frontiers)
        map_center = self.map_center.cpu().numpy() if torch.is_tensor(self.map_center) else np.asarray(self.map_center)
        select_pixels_world = (
            (select_pixels - np.array([[self.grid_dim[0] // 2, self.grid_dim[1] // 2]]))
            * float(self.cell_size)
            + map_center[None, :]
        )  # (N,2) [x,z]

        return select_pixels_world
    
    def visualize_map(self, c2w, world_goal_point = None, path = None, global_path = None):
        prob, index = self.occ_map.max(dim=0)
        map_center = self.map_center.cpu().numpy()

        # compute robot position
        cam_pos = c2w[:3, 3]
        cam_pos = cam_pos[[0, 2]] # switch to x,z
        cam_pos_x = (cam_pos[0] - map_center[0]) / self.cell_size + self.grid_dim[0] // 2
        cam_pos_z = (cam_pos[1] - map_center[1]) / self.cell_size + self.grid_dim[1] // 2

        index = index.cpu().numpy()
        grid_img = np.zeros((index.shape[0], index.shape[1], 3), dtype=np.uint8)

        for label in color_mapping_3.keys():
            # assign color based on the label
            grid_img[index == label] = color_mapping_3[label]

        grid_img = cv2.circle(grid_img, (int(cam_pos_x), int(cam_pos_z)), 5, (255, 0, 0), -1)

        # draw goal point
        if world_goal_point is not None:
            world_goal_point_x = (world_goal_point[0, 3] - map_center[0]) / self.cell_size + self.grid_dim[0] // 2
            world_goal_point_z = (world_goal_point[2, 3] - map_center[1]) / self.cell_size + self.grid_dim[1] // 2
            grid_img = cv2.circle(grid_img, (int(world_goal_point_x), int(world_goal_point_z)), 5, (255, 125, 0), -1)

        if hasattr(self, "target_frontier") and self.target_frontier is not None:
            grid_img[..., 0] = np.where(self.target_frontier, 0, grid_img[..., 0])
            grid_img[..., 1] = np.where(self.target_frontier, 255, grid_img[..., 1])
            grid_img[..., 2] = np.where(self.target_frontier, 255, grid_img[..., 2])

        if global_path is not None:
            for idx in range(len(global_path) - 1):
                p1 = global_path[idx]
                p2 = global_path[idx + 1]
                grid_img = cv2.line(grid_img, (p1[0], p1[1]), (p2[0], p2[1]), (191, 191, 64), 3)

        if path is not None:
            for idx in range(len(path) - 1):
                p1 = path[idx]
                p2 = path[idx + 1]
                grid_img = cv2.line(grid_img, (p1[0], p1[1]), (p2[0], p2[1]), (191, 64, 191), 1)

        # plt.figure()
        # plt.imshow(grid_img)
        # plt.savefig(os.path.join(self.eval_dir, "occ_{}.png".format(self.frame_idx)))

    def sample_random_candidate(self, agent_pos, free_space,
                                 sample_range = 1., sample_size:int = 100):
        """ Randomly sample candidate poses near the robot in range
        
        Args:
            agent_pos: (3, ) np.ndarray, the agent position
            free_space: (H, W) np.ndarray, the free space map; 1 - free
            sample_range: (float), the range to sample
            sample_size: (int), the number of samples
        Return:
            random_pose: (sample_size, 4, 4) torch.Tensor, the random poses
        """
        rng = np.random.default_rng()
        # random_pos = np.zeros((sample_size, 3))
        # random_pos[:, 0] = rng.uniform(agent_pos[0] - sample_range, agent_pos[0] + sample_range, (sample_size, ))
        # random_pos[:, 2] = rng.uniform(agent_pos[2] - sample_range, agent_pos[2] + sample_range, (sample_size, ))
        # random_pos[:, 1] = agent_pos[1]

        map_center = self.map_center.cpu().numpy()
        # candidate_map_coord_x = ((random_pos[:, 0] - map_center[0]) / self.cell_size + self.grid_dim[0] // 2).astype(np.int32)
        # candidate_map_coord_z = ((random_pos[:, 2] - map_center[1]) / self.cell_size + self.grid_dim[1] // 2).astype(np.int32)
        
        free_space_erode = cv2.erode(free_space.astype(np.uint8), np.ones((11, 11), np.uint8))
        # plt.figure()
        # plt.imshow(free_space_erode)
        # plt.savefig(os.path.join(self.eval_dir, "freespace_erode_{}.png".format(self.frame_idx)))
        # plt.close()

        map_coord_z, map_coord_x = np.where(free_space_erode == 1)
        world_coord_z = (map_coord_z + 0.5 - self.grid_dim[1] // 2) * self.cell_size + map_center[1]
        world_coord_x = (map_coord_x + 0.5 - self.grid_dim[0] // 2) * self.cell_size + map_center[0]
        
        # downsample 4x
        coord_idx = rng.choice(len(world_coord_z), len(world_coord_z) // 4)
        world_coord_z = world_coord_z[coord_idx]
        world_coord_x = world_coord_x[coord_idx]
        world_coord_y = np.ones_like(world_coord_z) * agent_pos[1]
        valid_pos = np.stack([world_coord_x, world_coord_y, world_coord_z], axis=1)

        random_angle = rng.uniform(0., 2 * np.pi, (len(valid_pos), ))
        random_quat = np.zeros((len(valid_pos), 4)) # (in w, x, y, z)
        random_quat[:, 0] = np.cos(random_angle / 2)
        random_quat[:, 2] = np.sin(random_angle / 2)

        random_quat = torch.from_numpy(random_quat).float().cuda()
        valid_pos = torch.from_numpy(valid_pos).float().cuda()
        
        random_pose = torch.zeros((len(valid_pos), 4, 4)).cuda()
        random_pose[:, :3, 3] = valid_pos
        random_pose[:, :3, :3] = build_rotation(random_quat)
        random_pose[:, 3, 3] = 1.

        random_pose[:, :, 1] *= -1
        random_pose[:, :, 2] *= -1

        return random_pose

    def pose_eval(self, poses, *args):
        num_pose = poses.shape[0]
        return torch.ones((num_pose, )), poses

    def global_planning(self, pose_evaluation_fn:Callable = None, gaussian_points = None, 
                        goal_proposal_fn:Callable = None, expansion=1, visualize=True, 
                        agent_pose=None, last_goal = None, slam=None):
        """ 
        Global Planning for next target goal 
        
        Args:
            pose_evaluation_fn: (Callable), calculate
            gaussian_points: 3D Gaussian means for navigation
            goal_proposal_fn: propose candidate pose when no frontiers
            expansion (int): expansion factor (increase when no best path is found)
            visualize:  
        """
        # build frontiers
        print(">> Global Planning")
        if self.frontier_select_method == "vlm":
            candidate_pos, free_space = self.build_vlm_frontiers(slam, gaussian_points)
        else:
            candidate_pos, free_space = self.build_frontiers(gaussian_points)
            # select_pixels_world, free_space = self.build_object_frontiers(gaussian_points)
            use_frontier = candidate_pos is not None

            # this is frontier mode, return directly
            if pose_evaluation_fn is None and not use_frontier:
                return None, None, None
    
        # generate random gaussians
        if self.add_random_gaussians:
            random_gaussian_params = self.generate_random_gaussians(candidate_pos)
        else:
            random_gaussian_params = None

        # propose goals when no frontiers exist
        if candidate_pos is None and goal_proposal_fn is not None:
            # propose goals
            candidate_pos = goal_proposal_fn(self.K, self.cam_height)

        # extract goals
        candidate_pose = []
        if candidate_pos is not None:
            # centering
            if isinstance(candidate_pos, np.ndarray):
                candidate_pos = torch.from_numpy(candidate_pos).cuda()
            if self.centering:
                candidate_pos = torch.mean(candidate_pos, dim=0, keepdim=True)

            # sample poses
            while len(candidate_pose) == 0:
                candidate_pose = self.generate_candidate(candidate_pos, expansion)
                # expand the radius
                expansion *= 1.5
            
                # select goals in freespace
                eroded_free_space = cv2.erode(free_space.astype(np.uint8), np.ones((10, 10), np.uint8))
                if eroded_free_space.sum() > 40 :
                    # when no frontiers 
                    candidate_xy = candidate_pose[:, [0, 2], 3]
                    candidate_xy[:, 0] = (candidate_xy[:, 0] - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2
                    candidate_xy[:, 1] = (candidate_xy[:, 1] - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2
                    candidate_xy = candidate_xy.long()

                    eroded_free_space = torch.from_numpy(eroded_free_space).cuda()
                    free_pose = eroded_free_space[candidate_xy[:, 1], candidate_xy[:, 0]]
                    candidate_pose = candidate_pose[free_pose]

        # add uniformly sampled poses
        if not use_frontier:
            # random sampling     
            random_pose = self.sample_random_candidate(agent_pose, free_space, sample_range=2 * expansion, sample_size=int(400*expansion))
            if len(candidate_pose) == 0:
                candidate_pose = random_pose
            else:
                candidate_pose = torch.cat([candidate_pose, random_pose], dim=0)

        # add last goal
        # if last_goal is not None:
        #     if isinstance(last_goal, np.ndarray):
        #         last_goal = torch.from_numpy(last_goal).float().cuda().unsqueeze(0)
        #     candidate_pose = torch.cat([candidate_pose, last_goal], dim=0)

        # append previous top candidates
        # if self.previous_candidates is not None:
        #     candidate_pose = torch.cat([candidate_pose, self.previous_candidates], dim=0)

        if pose_evaluation_fn is None:
            scores, poses = self.pose_eval(candidate_pose)
        else:
            scores, poses = pose_evaluation_fn(candidate_pose, random_gaussian_params)
        #visualize
        if visualize:
            occ_map = self.occ_map.argmax(0) == 1
            binarymap = occ_map.cpu().numpy().astype(np.uint8)
            
            # dilate binary map
            kernel = np.ones((3, 3), np.uint8)  
            binarymap = cv2.dilate(binarymap, kernel)

            #to RGB
            vis_map = np.zeros((binarymap.shape[0],binarymap.shape[1],3), np.uint8)
            vis_map[:,:,0][binarymap!=0] = 255
            vis_map[:,:,1][binarymap!=0] = 255
            vis_map[:,:,2][binarymap!=0] = 255

            #frontiers
            if self.frontier.sum() != 0:
                frontier = self.frontier.copy()
                frontier = cv2.dilate(frontier.astype(np.uint8), kernel, iterations=1) 
                vis_map[:,:,0][frontier!=0] = 0
                vis_map[:,:,1][frontier!=0] = 255
                vis_map[:,:,2][frontier!=0] = 0

            # candidate poses
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            for score, pose in zip(normalized_scores, poses):
                heatcolor = heatmap(score.item())[:3]
                pt = self.convert_to_map([pose[0,3],pose[2,3]])
                vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 1, (int(heatcolor[0]*255), int(heatcolor[1]*255), int(heatcolor[2]*255)), -1)
                # vis_map[pt[1],pt[0],:] = np.array([0,0,255])

            # agent position
            pt = self.convert_to_map([agent_pose[0],agent_pose[2]])
            vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 2, (255,0,0), -1)

            # for w_pose in select_pixels_world:
            #     cand = w_pose
            #     if isinstance(cand, torch.Tensor):
            #         cand = cand.detach().cpu().numpy()
            #     # prendi il primo punto se è (K,2/3)
            #     if cand.ndim == 2:
            #         cand = cand[0]
            #     # usa [x,z]
            #     if cand.shape[0] >= 3:
            #         cx, cz = float(cand[0]), float(cand[2])
            #     else:
            #         cx, cz = float(cand[0]), float(cand[1])

            #     cpt = self.convert_to_map([cx, cz])
            #     # marker magenta (cerchio pieno + bordo + croce)
            #     vis_map = cv2.circle(vis_map, (cpt[0], cpt[1]), 6, (255, 0, 255), -1)


            os.makedirs(os.path.join(self.eval_dir, "maps"), exist_ok=True)
            plt.imsave(os.path.join(self.eval_dir, "maps", "occmap_with_candidates_{}.png".format(self.frame_idx)), vis_map)
            plt.close()

            # plt.imsave(os.path.join(self.eval_dir, "occmap_with_candidates_{}.png".format(self.frame_idx)), vis_map)
            # plt.close()

        # and we only select the TOP 50 points
        topk = 20
        sort_index = torch.argsort(scores, descending=True)
        poses = poses[sort_index[:topk]]
        scores = scores[sort_index[:topk]]

        # log the topk candidates
        self.previous_candidates = poses
        # print("Poses: ", poses.shape, "Scores: ", scores.shape)
        return poses, scores, random_gaussian_params
    
    def visualize_occ_map(self, agent_pose=None, fronts=None):
        occ_map = self.occ_map.argmax(0) == 1
        binarymap = occ_map.cpu().numpy().astype(np.uint8)
        
        # dilate binary map
        kernel = np.ones((3, 3), np.uint8)  
        binarymap = cv2.dilate(binarymap, kernel)

        #to RGB
        vis_map = np.zeros((binarymap.shape[0],binarymap.shape[1],3), np.uint8)
        vis_map[:,:,0][binarymap!=0] = 255
        vis_map[:,:,1][binarymap!=0] = 255
        vis_map[:,:,2][binarymap!=0] = 255

    
        # agent position
        pt = self.convert_to_map([agent_pose[0],agent_pose[2]])
        vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 2, (255,0,0), -1)

        os.makedirs(os.path.join(self.eval_dir, "maps_known_env"), exist_ok=True)
        plt.imsave(os.path.join(self.eval_dir, "maps_known_env", "occmap_with_candidates_{}.png".format(self.frame_idx)), vis_map)
        plt.close()



    def global_planning_frontier(self, expansion=1, visualize=True, 
                        agent_pose=None, last_goal = None, slam=None):
        """ 
        Global Planning for next target goal 
        
        Args:
            expansion (int): expansion factor (increase when no best path is found)
            visualize:  
        """
        # build frontiers
        print(">> Generate possible candidate pose")
        candidate_pos, free_space = self.build_frontiers(None)

        # print("FRONTIERS BUILT: ", candidate_pos)
        use_frontier = candidate_pos is not None
        # this is frontier mode, return directly
        # if pose_evaluation_fn is None and not use_frontier:
        #     return None, None, None


        # generate random gaussians
        random_gaussian_params = None

        # # propose goals when no frontiers exist
        # if candidate_pos is None and goal_proposal_fn is not None:
        #     # propose goals
        #     candidate_pos = goal_proposal_fn(self.K, self.cam_height)

        # extract goals
        candidate_pose = []
        if candidate_pos is not None:
            # centering
            if isinstance(candidate_pos, np.ndarray):
                candidate_pos = torch.from_numpy(candidate_pos).cuda()
            if self.centering:
                candidate_pos = torch.mean(candidate_pos, dim=0, keepdim=True)

            # sample poses
            while len(candidate_pose) == 0:
                candidate_pose = self.generate_candidate(candidate_pos, expansion)
                # expand the radius
                expansion *= 1.5
            
                # select goals in freespace
                eroded_free_space = cv2.erode(free_space.astype(np.uint8), np.ones((10, 10), np.uint8))
                if eroded_free_space.sum() > 40 :
                    # when no frontiers 
                    candidate_xy = candidate_pose[:, [0, 2], 3]
                    candidate_xy[:, 0] = (candidate_xy[:, 0] - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2
                    candidate_xy[:, 1] = (candidate_xy[:, 1] - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2
                    candidate_xy = candidate_xy.long()

                    eroded_free_space = torch.from_numpy(eroded_free_space).cuda()
                    free_pose = eroded_free_space[candidate_xy[:, 1], candidate_xy[:, 0]]
                    candidate_pose = candidate_pose[free_pose]
            # print("Generate candidate time: ", time.time() - generate_candidate_time)
        # add uniformly sampled poses
        if not use_frontier:
            # random sampling     
            random_pose = self.sample_random_candidate(agent_pose, free_space, sample_range=2 * expansion, sample_size=int(400*expansion))
            if len(candidate_pose) == 0:
                candidate_pose = random_pose
            else:
                candidate_pose = torch.cat([candidate_pose, random_pose], dim=0)
        
        evaluate_time = time.time()
        scores, poses = self.pose_eval(candidate_pose)
        # print("Pose evaluation time: ", time.time() - evaluate_time)
        #visualize
        if visualize:
            occ_map = self.occ_map.argmax(0) == 1
            binarymap = occ_map.cpu().numpy().astype(np.uint8)
            
            # dilate binary map
            kernel = np.ones((3, 3), np.uint8)  
            binarymap = cv2.dilate(binarymap, kernel)

            #to RGB
            vis_map = np.zeros((binarymap.shape[0],binarymap.shape[1],3), np.uint8)
            vis_map[:,:,0][binarymap!=0] = 255
            vis_map[:,:,1][binarymap!=0] = 255
            vis_map[:,:,2][binarymap!=0] = 255

            #frontiers
            if self.frontier.sum() != 0:
                frontier = self.frontier.copy()
                frontier = cv2.dilate(frontier.astype(np.uint8), kernel, iterations=1) 
                vis_map[:,:,0][frontier!=0] = 0
                vis_map[:,:,1][frontier!=0] = 255
                vis_map[:,:,2][frontier!=0] = 0

            # candidate poses
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            for idx, (score, pose) in enumerate(zip(normalized_scores, poses)):
                pt = self.convert_to_map([pose[0, 3], pose[2, 3]])

                # if len(poses) == 1:
                    # Colore giallo (BGR): (0, 255, 255)
                vis_map = cv2.circle(vis_map, (pt[0], pt[1]), 2, (0, 255, 255), -1)
                # else:
                #     heatcolor = heatmap(score.item())[:3]
                #     color = (int(heatcolor[0] * 255), int(heatcolor[1] * 255), int(heatcolor[2] * 255))
                #     vis_map = cv2.circle(vis_map, (pt[0], pt[1]), 1, color, -1)
                # vis_map[pt[1],pt[0],:] = np.array([0,0,255])

            # agent position
            pt = self.convert_to_map([agent_pose[0],agent_pose[2]])
            vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 2, (255,0,0), -1)
            os.makedirs(os.path.join(self.eval_dir, "maps"), exist_ok=True)
            plt.imsave(os.path.join(self.eval_dir, "maps", "occmap_with_candidates_{}.png".format(self.frame_idx)), vis_map)
            plt.close()

        # print("Visualization time: ", time.time() - visualization_time)
        # and we only select the TOP 50 points
        topk = 20
        sort_index = torch.argsort(scores, descending=True)
        poses = poses[sort_index[:topk]]
        scores = scores[sort_index[:topk]]

        # log the topk candidates
        self.previous_candidates = poses

        return poses, scores, random_gaussian_params
    
    def global_object_planning(self, pose_evaluation_fn:Callable = None, gaussian_points = None, gaussian_points_scene=None, 
                        goal_proposal_fn:Callable = None, expansion=1, visualize=True, 
                        agent_pose=None, criterion=None):
        """ 
        Global Planning for next target goal 
        
        Args:
            pose_evaluation_fn: (Callable), calculate
            gaussian_points: 3D Gaussian means for navigation
            goal_proposal_fn: propose candidate pose when no frontiers
            expansion (int): expansion factor (increase when no best path is found)
            visualize:  
        """

        # build frontiers
        print(">> Global Object Planning")
        
        _, free_space = self.build_frontiers(gaussian_points_scene)
        candidate_obj_pos = self.build_object_frontiers(gaussian_points)
        use_frontier = candidate_obj_pos is not None

        # this is frontier mode, return directly
        if pose_evaluation_fn is None and not use_frontier:
            return None, None, None
    
        # generate random gaussians
        if self.add_random_gaussians:
            random_gaussian_params = self.generate_random_gaussians(candidate_obj_pos)
        else:
            random_gaussian_params = None

        random_gaussian_params=None
        
        # propose goals when no frontiers exist
        if candidate_obj_pos is None and goal_proposal_fn is not None:
            # propose goals
            candidate_obj_pos = goal_proposal_fn(self.K_object, self.cam_height)

        # extract goals
        candidate_pose = []
        if candidate_obj_pos is not None:
            # centering
            if isinstance(candidate_obj_pos, np.ndarray):
                candidate_obj_pos = torch.from_numpy(candidate_obj_pos).cuda()
            if self.centering:
                candidate_obj_pos = torch.mean(candidate_obj_pos, dim=0, keepdim=True)

            # sample poses
            while len(candidate_pose) == 0:
                candidate_pose = self.generate_candidate_adv_object(candidate_obj_pos, expansion, mode="sorted")
                # expand the radius
                expansion *= 1.5 
            
                # select goals in freespace
                eroded_free_space = cv2.erode(free_space.astype(np.uint8), np.ones((10, 10), np.uint8))
                if eroded_free_space.sum() > 40 :
                    # when no frontiers 
                    candidate_xy = candidate_pose[:, [0, 2], 3]
                    candidate_xy[:, 0] = (candidate_xy[:, 0] - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2
                    candidate_xy[:, 1] = (candidate_xy[:, 1] - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2
                    candidate_xy = candidate_xy.long()

                    eroded_free_space = torch.from_numpy(eroded_free_space).cuda()
                    free_pose = eroded_free_space[candidate_xy[:, 1], candidate_xy[:, 0]]
                    candidate_pose = candidate_pose[free_pose]

        # add uniformly sampled poses
        if not use_frontier:
            # random sampling     
            random_pose = self.sample_random_candidate(agent_pose, free_space, sample_range=2 * expansion, sample_size=int(400*expansion))
            if len(candidate_pose) == 0:
                candidate_pose = random_pose
            else:
                candidate_pose = torch.cat([candidate_pose, random_pose], dim=0)

        # add last goal
        # if last_goal is not None:
        #     if isinstance(last_goal, np.ndarray):
        #         last_goal = torch.from_numpy(last_goal).float().cuda().unsqueeze(0)
        #     candidate_pose = torch.cat([candidate_pose, last_goal], dim=0)

        # append previous top candidates
        # if self.previous_candidates is not None:
        #     candidate_pose = torch.cat([candidate_pose, self.previous_candidates], dim=0)

        if pose_evaluation_fn is None:
            scores, poses = self.pose_eval(candidate_pose)
        else:
            scores, poses = pose_evaluation_fn(candidate_pose, random_gaussian_params, criterion=criterion)
        
        #visualize
        if visualize:
            occ_map = self.occ_map.argmax(0) == 1
            binarymap = occ_map.cpu().numpy().astype(np.uint8)
            
            # dilate binary map
            kernel = np.ones((3, 3), np.uint8)  
            binarymap = cv2.dilate(binarymap, kernel)

            #to RGB
            vis_map = np.zeros((binarymap.shape[0],binarymap.shape[1],3), np.uint8)
            vis_map[:,:,0][binarymap!=0] = 255
            vis_map[:,:,1][binarymap!=0] = 255
            vis_map[:,:,2][binarymap!=0] = 255

            #frontiers
            if self.frontier.sum() != 0:
                frontier = self.frontier.copy()
                frontier = cv2.dilate(frontier.astype(np.uint8), kernel, iterations=1) 
                vis_map[:,:,0][frontier!=0] = 0
                vis_map[:,:,1][frontier!=0] = 255
                vis_map[:,:,2][frontier!=0] = 0

            
            if isinstance(scores, torch.Tensor):
                best_idx = int(torch.argmax(scores).item())
            else:
                best_idx = int(np.argmax(scores))

            best_pose = poses[best_idx]  # shape (4,4)

            # posizione in mondo -> pixel mappa
            best_pt = self.convert_to_map([best_pose[0, 3].item(), best_pose[2, 3].item()])

            cv2.circle(vis_map, (best_pt[0], best_pt[1]), 3, (0, 255, 255), -1)  # fill
            cv2.circle(vis_map, (best_pt[0], best_pt[1]), 4, (255, 255, 255), 2) 
            
            R = best_pose[:3, :3]
            # yaw coerente con il tuo codice (atan2(pose[0,2], pose[2,2]))
            yaw = np.arctan2(R[0, 2].item(), R[2, 2].item())

            # direzione in XZ (verso "avanti" della camera)
            dir_x = np.sin(yaw)
            dir_z = np.cos(yaw)

            # punto di partenza (pixel)
            start = (best_pt[0], best_pt[1])

            # lunghezza freccia in pixel (regola a piacere)
            arrow_len = 12

            # converti world->map per l’estremo della freccia
            end_world_x = best_pose[0, 3].item() + dir_x * (arrow_len * self.cell_size)
            end_world_z = best_pose[2, 3].item() + dir_z * (arrow_len * self.cell_size)
            end = self.convert_to_map([end_world_x, end_world_z])

            cv2.arrowedLine(vis_map, start, (end[0], end[1]), (0, 255, 255), 2, tipLength=0.35) 
            
            # candidate poses
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            for score, pose in zip(normalized_scores, poses):
                heatcolor = heatmap(score.item())[:3]
                pt = self.convert_to_map([pose[0,3],pose[2,3]])
                vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 1, (int(heatcolor[0]*255), int(heatcolor[1]*255), int(heatcolor[2]*255)), -1)
                # vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 1, (0, 255, 0), -1)
                # vis_map[pt[1],pt[0],:] = np.array([0,0,255])
            

            # cand = candidate_obj_pos
            # if isinstance(cand, torch.Tensor):
            #     cand = cand.detach().cpu().numpy()
            # # prendi il primo punto se è (K,2/3)
            # if cand.ndim == 2:
            #     cand = cand[0]
            # # usa [x,z]
            # if cand.shape[0] >= 3:
            #     cx, cz = float(cand[0]), float(cand[2])
            # else:
            #     cx, cz = float(cand[0]), float(cand[1])

            # cpt = self.convert_to_map([cx, cz])
            # # marker magenta (cerchio pieno + bordo + croce)
            # vis_map = cv2.circle(vis_map, (cpt[0], cpt[1]), 4, (0, 255, 255), 2)


            # agent position
            pt = self.convert_to_map([agent_pose[0],agent_pose[2]])
            cv2.circle(vis_map, (pt[0], pt[1]), 4, (255, 0, 0), 2) 
            vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 2, (255,0,0), -1)
            
            os.makedirs(os.path.join(self.eval_dir, "maps"), exist_ok=True)
            plt.imsave(os.path.join(self.eval_dir, "maps", "occmap_with_candidates_{}.png".format(self.frame_idx)), vis_map)
            plt.close()
            # plt.imsave(os.path.join(self.eval_dir, "occmap_with_candidates_{}.png".format(self.frame_idx)), vis_map)
            # plt.close()

        # and we only select the TOP 50 points
        topk = 20
        sort_index = torch.argsort(scores, descending=True)
        poses = poses[sort_index[:topk]]
        scores = scores[sort_index[:topk]]

        # log the topk candidates
        self.previous_candidates = poses
        # print("Poses: ", poses.shape, "Scores: ", scores.shape)
        return poses, scores, random_gaussian_params, candidate_obj_pos
    
    def generate_random_gaussians(self, candidate_pos):
        """ Generate Random Gaussians from candidate positions """
        if candidate_pos is None:
            return None

        GAUSSIAN_PER_GRID = 200

        position = torch.from_numpy(candidate_pos).float().cuda()
        xz_offset = torch.rand((1, GAUSSIAN_PER_GRID, 2)).float().cuda() * self.cell_size
        y_offset = (self.cam_height - 1.0) + torch.rand((candidate_pos.shape[0], GAUSSIAN_PER_GRID, 1)).float().cuda()

        new_p3t = torch.cat([position[:, None, :] + xz_offset, y_offset], dim=-1)
        new_p3t = new_p3t.reshape(-1, 3)
        new_p3t = new_p3t[:, [0, 2, 1]] # change to x-y-z order

        shs = torch.rand((new_p3t.shape[0], 1, 3)).float().cuda()
        new_rotations = torch.zeros((new_p3t.shape[0], 4)).float().cuda()
        new_rotations[:, 0] = 1.
        new_opacities = torch.rand((new_p3t.shape[0], 1)).float().cuda().clamp_(min=1e-3)
        new_scales = torch.rand((new_p3t.shape[0], 3)).float().cuda().clamp_(min=1e-3) * self.cell_size * 0.05

        gaussian_params = dict(means3D=new_p3t, scales=new_scales, rotations=new_rotations, opacity=new_opacities, shs=shs)
        return gaussian_params

    def convert_to_map(self, coord):
        map_center = self.map_center.cpu().numpy()
        cam_pos_x = int((coord[0] - map_center[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((coord[1] - map_center[1]) / self.cell_size + self.grid_dim[1] // 2)
        return np.array([cam_pos_x, cam_pos_z])
    
    def convert_to_world(self, coord):
        map_center = self.map_center.cpu().numpy()
        world_coord = (coord - self.grid_dim / 2) * self.cell_size + map_center
        return world_coord

    def generate_candidate(self, center_point:torch.Tensor, expansion=1):
        """ 
        sample camera poses from the center point, 
        Args:
            center_point: (K, 3) tensor, the local from which camera poses are sampled
        """

        K, radius = self.K, self.radius * expansion
        # radius = min( radius * (self.selection + 1), 5 )
        theta = torch.rand((K, )).cuda() * 2 * torch.pi
        random_radius = self.min_range + torch.rand((K, )).cuda() * (radius - self.min_range)

        # sample K points from center point with replacement.
        center_point_height = torch.ones((center_point.shape[0], ), device=center_point.device) * self.cam_height
        center_point = torch.stack([center_point[:, 0], center_point_height, center_point[:, 1]], dim=1)
        center_point_rand_index = torch.randint(0, center_point.shape[0], (K, ))
        center_point = center_point[center_point_rand_index]

        cam_pos = torch.zeros((K, 3)).cuda()
        cam_pos[:, 0] = center_point[:, 0] + random_radius * torch.sin(theta)
        cam_pos[:, 1] = self.cam_height # keep the same height
        cam_pos[:, 2] = center_point[:, 2] + random_radius * torch.cos(theta)

        # Random generate camera rotation
        cam_rot = torch.zeros((K, 4)).cuda()
        
        # theta = torch.atan2(-random_offset[:, 0], -random_offset[:, 2])
        theta = theta + torch.pi
        
        cam_rot[:, 0] = torch.cos(theta / 2)
        cam_rot[:, 2] = torch.sin(theta / 2)
        cam_R = build_rotation(cam_rot)

        # Rotate along z-axis to let y-axis facing downward.
        cam_R[:, :, 0] *= -1
        cam_R[:, :, 1] *= -1

        c2ws = torch.zeros((K, 4, 4)).cuda()
        c2ws[:, :3, 3] = cam_pos
        c2ws[:, :3, :3] = cam_R
        c2ws[:, 3, 3] = 1.

        return c2ws

    def generate_candidate_object(self, center_point:torch.Tensor, expansion=1):
        """ 
        sample camera poses from the center point, 
        Args:
            center_point: (K, 3) tensor, the local from which camera poses are sampled
        """

        K, radius = self.K_object, self.radius_object * expansion
        # radius = min( radius * (self.selection + 1), 5 )
        theta = torch.rand((K, )).cuda() * 2 * torch.pi
        random_radius = self.min_range_object + torch.rand((K, )).cuda() * (radius - self.min_range_object)

        # sample K points from center point with replacement.
        center_point_height = torch.ones((center_point.shape[0], ), device=center_point.device) * self.cam_height
        center_point = torch.stack([center_point[:, 0], center_point_height, center_point[:, 1]], dim=1)
        center_point_rand_index = torch.randint(0, center_point.shape[0], (K, ))
        center_point = center_point[center_point_rand_index]

        cam_pos = torch.zeros((K, 3)).cuda()
        cam_pos[:, 0] = center_point[:, 0] + random_radius * torch.sin(theta)
        cam_pos[:, 1] = self.cam_height # keep the same height
        cam_pos[:, 2] = center_point[:, 2] + random_radius * torch.cos(theta)

        # Random generate camera rotation
        cam_rot = torch.zeros((K, 4)).cuda()
        
        # theta = torch.atan2(-random_offset[:, 0], -random_offset[:, 2])
        theta = theta + torch.pi
        
        cam_rot[:, 0] = torch.cos(theta / 2)
        cam_rot[:, 2] = torch.sin(theta / 2)
        cam_R = build_rotation(cam_rot)

        # Rotate along z-axis to let y-axis facing downward.
        cam_R[:, :, 0] *= -1
        cam_R[:, :, 1] *= -1

        c2ws = torch.zeros((K, 4, 4)).cuda()
        c2ws[:, :3, 3] = cam_pos
        c2ws[:, :3, :3] = cam_R
        c2ws[:, 3, 3] = 1.

        return c2ws
    
    def generate_candidate_adv_object(self,
                              center_point: torch.Tensor,
                              expansion: float = 1,
                              mode: str = "random",             # "random" (default) | "sorted"
                              theta_step_deg: float = 15.0,      # passo angolare per "sorted"
                              radial_bins: int = 6,              # # anelli per "sorted"
                              radial_spacing: str = "linear"     # "linear" | "sqrt_area"
                              ):
        """ 
        Sample camera poses around the (one or more) center points.
        - mode="random": comportamento originale (campionamento casuale su anello).
        - mode="sorted": griglia ordinata (angoli equispaziati + raggi discreti).
        
        Args:
            center_point: (Kc, 3) tensor — punti attorno a cui campionare (x,z usati; y forzato a cam_height)
            expansion: fattore che scala il raggio massimo
            theta_step_deg: ampiezza passo angolare per la modalità "sorted"
            radial_bins: numero di anelli radiali in "sorted"
            radial_spacing: "linear" (r lineare) o "sqrt_area" (uniforme in area dell’anello)
        """
        device = center_point.device
        K, radius = self.K_object, self.radius_object * expansion

        # ---- prepara i center_point a quota fissa (come nel tuo codice) ----
        center_point_height = torch.ones((center_point.shape[0],), device=device) * self.cam_height
        # reinterpreta (x, z) dal center_point[:,0], center_point[:,1] -> [x, cam_height, z]
        center_point = torch.stack([center_point[:, 0], center_point_height, center_point[:, 1]], dim=1)

        # campiona K centri con replacement (comportamento invariato)
        center_point_rand_index = torch.randint(0, center_point.shape[0], (K,), device=device)
        center_point = center_point[center_point_rand_index]  # (K, 3)

        # ---- campionamento posizioni camera ----
        if mode.lower() == "random":
            # === COMPORTAMENTO ORIGINALE ===
            theta = torch.rand((K,), device=device) * 2 * torch.pi
            random_radius = self.min_range_object + torch.rand((K,), device=device) * (radius - self.min_range_object)

            cam_pos = torch.zeros((K, 3), device=device)
            cam_pos[:, 0] = center_point[:, 0] + random_radius * torch.sin(theta)
            cam_pos[:, 1] = self.cam_height
            cam_pos[:, 2] = center_point[:, 2] + random_radius * torch.cos(theta)

        elif mode.lower() == "sorted":
            # === GRIGLIA ORDINATA: anelli radiali + angoli equispaziati ===
            # 1) angoli
            step_rad = torch.deg2rad(torch.tensor(theta_step_deg, device=device))
            # almeno 1 campione
            num_theta = max(1, int(torch.round(2 * torch.pi / step_rad).item()))
            thetas = torch.linspace(0.0, 2 * torch.pi, steps=num_theta, device=device, dtype=torch.float32)
            # l'ultimo coincide col primo; meglio escludere l’estremo per evitare duplicati
            thetas = thetas[:-1] if thetas.numel() > 1 else thetas

            # 2) raggi (radial_bins anelli da min_range a radius)
            radial_bins = max(1, int(radial_bins))
            if radial_spacing == "sqrt_area" and radial_bins > 1:
                # uniforme in area dell’anello: r^2 lineare
                u = torch.linspace(0.0, 1.0, steps=radial_bins, device=device)
                r_min2 = self.min_range_object ** 2
                r_max2 = radius ** 2
                r_vals = torch.sqrt(r_min2 + u * (r_max2 - r_min2))
            else:
                # lineare semplice
                r_vals = torch.linspace(self.min_range_object, radius, steps=radial_bins, device=device)

            # 3) mesh (r, theta) → lista di candidate
            R, T = torch.meshgrid(r_vals, thetas, indexing='ij')  # (radial_bins, num_theta)
            R = R.reshape(-1)  # (radial_bins*num_theta,)
            T = T.reshape(-1)

            # 4) costruiamo posizioni; dobbiamo produrre esattamente K pose
            #    se la griglia ha più/meno elementi di K, facciamo slice o repeat
            total = R.numel()
            if total < K:
                rep = (K + total - 1) // total
                R = R.repeat(rep)[:K]
                T = T.repeat(rep)[:K]
            elif total > K:
                R = R[:K]
                T = T[:K]

            # usa i center_point "accoppiati" (già di dimensione K) come nel random
            cam_pos = torch.zeros((K, 3), device=device)
            cam_pos[:, 0] = center_point[:, 0] + R * torch.sin(T)
            cam_pos[:, 1] = self.cam_height
            cam_pos[:, 2] = center_point[:, 2] + R * torch.cos(T)

            # riusa T come yaw di base (che punta radialmente verso l’esterno); poi lo ruotiamo di π per guardare il centro
            theta = T
        else:
            raise ValueError("mode must be 'random' or 'sorted'")

        # ---- orientamento (look-at verso il centro) identico al tuo ----
        # se in random avevamo theta random, in sorted abbiamo T; in ogni caso aggiungiamo π
        # per orientare lo sguardo verso il centro
        if mode.lower() == "random":
            # theta definito nella branch random
            yaw = theta + torch.pi
        else:
            # theta = T nella branch sorted
            yaw = theta + torch.pi

        cam_rot = torch.zeros((K, 4), device=device)
        cam_rot[:, 0] = torch.cos(yaw / 2)  # w
        cam_rot[:, 2] = torch.sin(yaw / 2)  # y  (assumendo convenzione [w, x, y, z] con yaw su asse Y)
        cam_R = build_rotation(cam_rot)

        # flip X e Y come nel tuo codice (convenzione camera)
        cam_R[:, :, 0] *= -1
        cam_R[:, :, 1] *= -1

        # ---- composizione c2w ----
        c2ws = torch.zeros((K, 4, 4), device=device)
        c2ws[:, :3, 3] = cam_pos
        c2ws[:, :3, :3] = cam_R
        c2ws[:, 3, 3] = 1.0

        return c2ws


    def planning(self, goal):
        # check goal first
        # print(">> Planning for goal: ", goal)
        if self.occ_map_np[goal[0], goal[1]]:
            print("Goal is occupied, cannot plan.")
            return np.array([])
        
        dist_to_goal_map = np.ones_like(self.occ_map_np, dtype=np.float32) * -1

        visited_node_y, visited_node_x = np.where(self.planning_direction[..., 1] >= 0)
        distance_to_goal = np.linalg.norm([visited_node_y - goal[0], visited_node_x - goal[1]], axis=0)
        dist_to_goal_map[visited_node_y, visited_node_x] = distance_to_goal

        # build selection frontiers
        # get the free space, constructed from setup_start function
        free_space = self.free_space_np

        searched_area = self.planning_direction[..., 1] >= 0 # 1 - searched; 0 - not searched
        searched_area = searched_area.astype(np.uint8)

        distance_to_obstacle = cv2.distanceTransform(free_space, cv2.DIST_L1, 5)

        # search frontiers
        frontiers =  searched_area - cv2.erode(searched_area, np.ones((3, 3), np.uint8))
        frontiers = frontiers * free_space
        frontiers_y, frontiers_x = np.where(frontiers > 0)
        frontiers = np.stack([frontiers_y, frontiers_x], axis=1)
        dist_to_goal_map[frontiers_y, frontiers_x] = np.linalg.norm([frontiers_y - goal[0], frontiers_x - goal[1]], axis=0)

        frontiers = [(dist_to_goal_map[point[0], point[1]], point[0], point[1]) for point in frontiers]
        heapq.heapify(frontiers)

        max_iter = 1e4
        curr_iter = 0
        # A star searching
        # print(">> Start A* search")
        while curr_iter < max_iter:
            if len(frontiers) == 0:
                break
            # find the node with the smallest cost
            dist_to_goal, current_node_y, current_node_x = heapq.heappop(frontiers)
            current_node = np.array([current_node_y, current_node_x])

            if np.max(abs(current_node - goal)) < 2: # current_node[0] == goal[0] and current_node[1] == goal[1]:
                goal = current_node
                break
            
            current_y, current_x = current_node[0], current_node[1]

            # search the 8-neighbors
            neighbors = np.array([
                [current_y - 3, current_x], 
                [current_y - 3, current_x + 1], 
                [current_y - 3, current_x + 3], 
                [current_y - 1, current_x + 3], 
                [current_y, current_x + 3], 

                [current_y + 3, current_x], 
                [current_y + 3, current_x + 1], 
                [current_y + 3, current_x + 3], 
                [current_y + 1, current_x + 3], 

                [current_y - 3, current_x - 1], 
                [current_y - 3, current_x - 3], 
                [current_y - 1, current_x - 3], 
                [current_y, current_x - 3], 

                [current_y + 3, current_x - 1], 
                [current_y + 3, current_x - 3], 
                [current_y + 1, current_x - 3]])
            
            neighbor_path = np.array([
                [[current_y - 1, current_x], [current_y - 2, current_x], [current_y - 3, current_x]],
                [[current_y - 1, current_x], [current_y - 2, current_x + 1],  [current_y - 3, current_x + 1]],
                [[current_y - 1, current_x + 1], [current_y - 2, current_x + 2], [current_y - 3, current_x + 3]],
                [[current_y, current_x + 1], [current_y - 1, current_x + 2],  [current_y - 1, current_x + 3]],
                [[current_y, current_x + 1], [current_y, current_x + 2],  [current_y, current_x + 3]],

                [[current_y + 1, current_x], [current_y + 2, current_x], [current_y + 3, current_x]],
                [[current_y + 1, current_x], [current_y + 2, current_x + 1],  [current_y + 3, current_x + 1]],
                [[current_y + 1, current_x + 1], [current_y + 2, current_x + 2], [current_y + 3, current_x + 3]],
                [[current_y, current_x + 1], [current_y + 1, current_x + 2],  [current_y + 1, current_x + 3]],

                [[current_y - 1, current_x], [current_y - 2, current_x - 1],  [current_y - 3, current_x - 1]],
                [[current_y - 1, current_x - 1], [current_y - 2, current_x - 2], [current_y - 3, current_x - 3]],
                [[current_y, current_x - 1], [current_y - 1, current_x - 2],  [current_y - 1, current_x - 3]],
                [[current_y, current_x - 1], [current_y, current_x - 2], [current_y, current_x - 3]],

                [[current_y + 1, current_x], [current_y + 2, current_x - 1],  [current_y + 3, current_x - 1]],
                [[current_y + 1, current_x - 1], [current_y + 2, current_x - 2], [current_y + 3, current_x - 3]],
                [[current_y, current_x - 1], [current_y + 1, current_x - 2],  [current_y + 1, current_x - 3]],
            ])

            x_width_right, x_width_left = neighbor_path[:9] + np.array([[[0, 1]]]), neighbor_path[:9] + np.array([[[0, -1]]])
            y_width_up, y_width_down = neighbor_path[9:] + np.array([[[1, 0]]]), neighbor_path[9:] + np.array([[[-1, 0]]])
            road_width_left, road_width_right = np.concatenate([x_width_right, y_width_up], axis=0), np.concatenate([x_width_left, y_width_down], axis=0)
            neighbor_path = np.concatenate([neighbor_path, road_width_left, road_width_right], axis=1) # (K, 9, 2)

            inside = (neighbor_path[:, :, 0] >= 0) & (neighbor_path[:, :, 0] < self.occ_map_np.shape[0]) &  (neighbor_path[:, :, 1] >= 0) & (neighbor_path[:, :, 1] < self.occ_map_np.shape[1])
            inside = np.all(inside, axis=1)

            # remove the neighbors out of the map
            neighbors = neighbors[inside]
            neighbor_path = neighbor_path[inside]
            
            # remove the neighbors that are not free space
            neighbor_path_ = neighbor_path.reshape(-1, 2)
            free_path = free_space[neighbor_path_[:, 0], neighbor_path_[:, 1]]
            free_path = free_path.reshape(-1, 9)
            free_path = np.all(free_path, axis=1)
            
            neighbors = neighbors[free_path]
            neighbor_path = neighbor_path[free_path]
            
            for neighbor, neighbor_path in zip(neighbors, neighbor_path):
                dist_obs = distance_to_obstacle[neighbor_path[:, 0], neighbor_path[:, 1]]
                collision_cost_array = np.zeros_like(dist_obs)

                collision_cost_array[dist_obs > 20] = 0
                collision_cost_array[np.bitwise_and(dist_obs > 10, dist_obs <= 20)] = 4
                collision_cost_array[np.bitwise_and(dist_obs > 5, dist_obs <= 10)] = 8
                collision_cost_array[dist_obs <= 5] = 12

                dist_cost = self.planning_direction[current_node[0], current_node[1], 0] \
                            + np.linalg.norm(neighbor - current_node)
                collision_cost = self.planning_direction[current_node[0], current_node[1], 3] \
                            + collision_cost_array.sum()
                
                if self.planning_direction[neighbor[0], neighbor[1], 0] < 0 or \
                    self.planning_direction[neighbor[0], neighbor[1], 0] + self.planning_direction[neighbor[0], neighbor[1], 3] > dist_cost + collision_cost:
                    self.planning_direction[neighbor[0], neighbor[1], 0] = dist_cost
                    self.planning_direction[neighbor[0], neighbor[1], 1] = current_node[0]
                    self.planning_direction[neighbor[0], neighbor[1], 2] = current_node[1]
                    self.planning_direction[neighbor[0], neighbor[1], 3] = collision_cost
                    dist_to_goal_map[neighbor[0], neighbor[1]] = np.linalg.norm(neighbor - goal)
                    
                    heapq.heappush(frontiers, (dist_to_goal_map[neighbor[0], neighbor[1]] + collision_cost, neighbor[0], neighbor[1]))

            curr_iter += 1

        # No valid path
        if self.planning_direction[goal[0], goal[1], 0] < 0:
            print("No valid path found to the goal.")
            return np.array([])

        # generate path
        path = [goal]
        while True:

            parent = self.planning_direction[path[-1][0], path[-1][1], 1:].astype(np.int32)
            if parent[0] == path[-1][0] and parent[1] == path[-1][1]:
                # print("Reached the start point.")
                break
            path.append([parent[0], parent[1]])
        
        if len(path) == 1:
            return np.array([])

        # take the reverse order and change to x-z order
        paths = np.array(path)
        paths = paths[::-1]
        paths = paths[:, [1, 0]] # x-z

        # # create shorcut path by removing the intermediate points
        if self.shortcut_path:
            shortcut_path = [paths[0], paths[1]]
            path_idx = 1
            for i in range(2, paths.shape[0] - 1):
                shortcut = self.CheckCollision(shortcut_path[path_idx - 1], paths[i], self.occ_map_np)

                # if free path from path_idx - 1
                if shortcut:
                    shortcut_path[path_idx] = paths[i]
                else:
                    shortcut_path.append(paths[i])
                    path_idx += 1

            # # add the goal point
            shortcut_path.append(paths[-1])
            paths = np.stack(shortcut_path, axis=0)
        # print("Path found: ", paths.shape)
        return paths

    def CheckCollision(self, pt1, pt2, occ_map):
        traj = np.zeros_like(occ_map)
        traj = cv2.line(traj, tuple(pt1), tuple(pt2), 1, 7)
        return np.all(occ_map[traj == 1] == 0)
    
    def render_bev(self, slam):
        bev_c2w = torch.tensor([[1., 0., 0., 0.],
                        [0., 0., -1., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.]]).float().cuda()
        bev_c2w[:3, 3] = torch.tensor([self.map_center[0], 7., self.map_center[1]]).cuda()
        xyz = slam.get_gaussian_xyz()
        bev_mask = xyz[:, 1] < self.cam_height # cam height is 1.5
        bev_render_pkg = slam.render_at_pose(bev_c2w.cuda(), white_bg=True, mask=bev_mask)
        return bev_render_pkg
    
    def occ_coord_to_3d(self, occ_coord):
        """ Convert occ_map coordinate to 3D coordinate """
        map_center = self.map_center.cpu().numpy()
        frontier_pts = occ_coord[:, [1, 0]] # switch to x,z as in build_frontier
        frontier_pts = (frontier_pts - np.array([[self.grid_dim[0] // 2, self.grid_dim[1] // 2]])) * self.cell_size + map_center[None, :]
        frontier_pts3d = np.zeros((frontier_pts.shape[0], 3))
        frontier_pts3d[:, [0, 2]] = frontier_pts
        frontier_pts3d[:, 1] = self.cam_height
        return frontier_pts3d

    def get_map(self):
        return self.occ_map