# NOTE: This is codebase from SplaTAM *NOT* GuassianSLAM aka MonoGS
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time
import open3d as o3d
import matplotlib.pyplot as plt
import os
import yaml
from typing import Optional, List, Tuple, Dict

from sklearn.cluster import DBSCAN

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from models.SLAM.utils.recon_helpers import setup_camera
from models.SLAM.utils.common_utils import seed_everything, save_params_ckpt, save_params
from models.SLAM.utils.eval_helpers import report_loss, report_progress, eval
from models.SLAM.utils.keyframe_selection import keyframe_selection_overlap
from models.SLAM.utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion, calc_loss
)
from models.SLAM.utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify, compute_next_campos, remove_points
import datasets.util.map_utils as map_utils
import habitat_sim
import cv2

import wandb as wandb_run
from einops import repeat, rearrange, reduce
from scipy.spatial.transform import Rotation 
from models.SLAM.droid_wrapper import DroidWrapper
from yacs.config import CfgNode
import datasets.util.utils as utils
from cluster_manager import ClusterStateManager

import logging 
logger = logging.getLogger("rich")

color_mapping_3 = {
    0:np.array([255,255,255]), # white
    1:np.array([0,0,255]), # blue
    2:np.array([0,255,0]), # green
}

def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, downsample=1, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    """
    Un-project all points from rgbd data to get point cloud. 
    Return:
        point_cld (N, 6) (xyz+rgb)
        mean3_sq_dist (N, ) distance of each gaussian
    """
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(0, width, step=downsample).cuda().float(), 
                                    torch.arange(0, height, step=downsample).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0, ::downsample, ::downsample].reshape(-1)

    # Initialize point cloud
    # For habitat, the camera looks at -Z
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(int(height * width / (downsample ** 2)), 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c).cuda()
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = downsample * depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    # downsample color image
    downsampled_color = color[:, ::downsample, ::downsample]
    cols = torch.permute(downsampled_color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        downsampled_mask = mask.reshape(color.shape[1], color.shape[2])
        # downsampled_mask = downsampled_mask[::downsample, ::downsample].reshape(-1)
        downsampled_mask = F.max_pool2d(downsampled_mask.unsqueeze(0).float(), downsample).bool()
        downsampled_mask = rearrange(downsampled_mask, "b h w -> (b h w)") 
        if downsampled_mask.sum() > 0:
            point_cld = point_cld[downsampled_mask]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[downsampled_mask]
        else:
            print("[WARN] Mask become all zero after downsampling")

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, w2c = None, isotropic=False):
    """
    Initialize Gaussians Parameters
        params: dict
            cam_unnorm_rots: quat (wxyz) first frame as identity
            cam_trans: xyz 
    """
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3] - in w, x, y, z
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if isotropic else 3)),
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables

@torch.enable_grad()
def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1,ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    
    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                                gaussians_grad=True,
                                                camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    color_mask = torch.tile(mask, (3, 1, 1))
    color_mask = color_mask.detach()

    losses = calc_loss(curr_data, im, depth, mask, color_mask, 
                        use_l1, use_sil_for_loss, ignore_outlier_depth_loss, tracking)

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss and (tracking_iteration + 1) % 10 == 0:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(curr_data['im'].cpu().permute(1, 2, 0))
        ax[1, 3].set_title("GT Image")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        # plt.close()
        # plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        # cv2.imshow('Diff Images', plot_img)
        # cv2.waitKey(1)

        ## Save Tracking Loss Viz
        save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        os.makedirs(save_plot_dir, exist_ok=True)
        plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses

def initialize_new_params(new_pt_cld, mean3_sq_dist, isotropic=False):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]                       # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3 if not isotropic else 1)),
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params

def add_new_gaussians(config, params, variables, curr_data, sil_thres, 
                    time_idx, mean_sq_dist_method, densify_dict, 
                    add_rand_gaussians = True, downsample_pcd = 1):
    """ Add new gaussians based """
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > densify_dict["depth_error_ratio"] * depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)
    # Filter Depth
    valid_depth_mask = (curr_data['depth'][0, :, :] > 0.01)
    non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran

        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True, downsample=downsample_pcd,
                                    mean_sq_dist_method=mean_sq_dist_method)
        
        # # add gaussians slightly backward
        # new_pt_cld_back, mean3_sq_dist_back = get_pointcloud(curr_data['im'], curr_data['depth'] + 0.01, curr_data['intrinsics'], 
        #                             curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True, downsample=4,
        #                             mean_sq_dist_method=mean_sq_dist_method)
        
        # new_pt_cld  = torch.cat([new_pt_cld, new_pt_cld_back], dim=0)
        # mean3_sq_dist = torch.cat([mean3_sq_dist, mean3_sq_dist_back], dim=0)
       
        if add_rand_gaussians:
            # # add random gaussians to new_pt_cld
            num_pts = int(min(params["means3D"].shape[0], 1e2))
            scene_boundary_max, scene_boundary_min = torch.max(params["means3D"], dim=0)[0], torch.min(params["means3D"], dim=0)[0]
            new_boundary_max, new_boundary_min = torch.max(new_pt_cld[:, :3], dim=0)[0], torch.min(new_pt_cld[:, :3], dim=0)[0]
            scene_boundary_max, scene_boundary_min = torch.maximum(scene_boundary_max, new_boundary_max), torch.minimum(scene_boundary_min, new_boundary_min)
            extent = (scene_boundary_max - scene_boundary_min) / 2
            center = (scene_boundary_max + scene_boundary_min) / 2

            center[1] = params["cam_trans"][0, 1, 0].item() # robot height
            extent[1] = 1.0

            # add random gaussians into unkown region.
            

            # random_seed_y = torch.rand((num_pts, ), device=new_pt_cld.device) - 0.5
            random_seed = torch.rand((num_pts * 2, 3), device=new_pt_cld.device) * 2 - 1 # [-1, 1]
            inside = torch.bitwise_and(
                torch.bitwise_and(-0.8 <= random_seed[:, 0],  random_seed[:, 0]<= 0.8), 
                torch.bitwise_and(-0.8 <= random_seed[:, 2],  random_seed[:, 2]<= 0.8)
            )
            random_seed = random_seed[~inside]
            random_seed[:, 1] = torch.rand((len(random_seed), ), device=new_pt_cld.device) - 0.5

            new_points =  random_seed * extent + center
            colors = torch.rand((len(random_seed), 3), device=new_pt_cld.device)
            scales = torch.ones((len(random_seed), ), device=new_pt_cld.device) * .5
            if len(new_pt_cld) > 0:
                new_pt_cld  = torch.cat([new_pt_cld, torch.cat([new_points, colors], dim=-1)], dim=0)
                mean3_sq_dist = torch.cat([mean3_sq_dist, scales], dim=0)
            else:
                new_pt_cld = torch.cat([new_points, colors], dim=-1)
                mean3_sq_dist = scales


        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, isotropic=config["isotropic"])
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables

class GaussianSLAM:
    # config = dict()

    def __init__(self, config: CfgNode):
        """
        params:
            config
        """
        # Camera intrinsics
        calibration = config["SLAM"]["Dataset"]["Calibration"]
        # Camera prameters
        width = calibration["width"]
        height = calibration["height"]
        K = np.array(
            [[calibration["fx"], 0.0, calibration["cx"]], [0.0, calibration["fy"], calibration["cy"]], [0.0, 0.0, 1.0]]
        )

        self.intrinsics = torch.from_numpy(K).float().cuda()

        # # Load config
        # with open(config, 'r') as f:
        #     config_dict = yaml.load(f, Loader=yaml.FullLoader)
        # self.config.update(config_dict)
        self.config = config
        self.cfg = config
    
        # Gaussian parameters
        self.params = {}
        self.variables = {}
        self.cam = None
        self.eval_dir = os.path.join(self.config["workdir"], self.config["run_name"])
        self.save_dir = self.eval_dir
        self.frame_idx = 0
        os.makedirs(self.eval_dir, exist_ok=True)

        # We don't need to save config again as itg has been saved in the tester.py

        # lrs dict 
        self.lrs_dict = {}

        self.initialize = False
        self.scene_radius_depth_ratio = 1.

        self.gt_w2c_all_frames = []
        self.keyframe_list = []
        self.keyframe_time_indices = []

        self.win_size = 10
        self.frames = []
        self.scorePoints = None
        self.target_fronter = None
        self.selection = 0

        self.droid = None
        self.frontier = None

        self.sm = ClusterStateManager()
        self.cell_size = self.config["explore"]["cell_size"]

    def init(self, color: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor,
             scene_bounds: Optional[List[np.ndarray]] = None):
        """
        Initialize the SLAM system
        Params:
            rgb (H, W, 3) in range (0, 255)
            depth (H, W, 1)
            pose (4, 4) c2w
        """
        # Get RGB-D Data & Camera Parameters
        # color, depth, intrinsics, pose = dataset[0]

        # Process RGB-D Data
        color = color.permute(2, 0, 1).float() / 255 # (H, W, C) -> (C, H, W)
        
        # Process Camera Parameters
        intrinsics = self.intrinsics # [:3, :3]
        w2c = torch.linalg.inv(pose)

        # manually set the view matrix to identity
        self.first_frame_w2c = torch.eye(4).cuda()
        # Setup Camera
        self.cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), np.eye(4))

        # [fx, fy, cx, cy]
        _, h0, w0 = color.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        densify_intrinsics = intrinsics

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        mask = (depth > 10 * self.cell_size) # Mask out invalid depth values
        mask = mask.reshape(-1)
        init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                    mask=mask, compute_mean_sq_dist=True, downsample = self.config["downsample_pcd"],
                                                    mean_sq_dist_method="projective")

        # Initialize Parameters
        self.params, self.variables = initialize_params(init_pt_cld, self.config["num_frames"], 
                                                            mean3_sq_dist, isotropic=self.config["isotropic"])

        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        self.variables['scene_radius'] = torch.max(depth) / self.config["scene_radius_depth_ratio"]

        # Set the first frame to w2c
        rel_w2c_rot = w2c[:3, :3].unsqueeze(0).detach()
        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
        rel_w2c_tran = w2c[:3, 3].detach()

        # Update the camera parameters
        self.params['cam_unnorm_rots'][..., 0] = rel_w2c_rot_quat
        self.params['cam_trans'][..., 0]= rel_w2c_tran

        # The first frame begins after initilization
        self.frame_idx = 0
        self.initialize = True
        self.cam_height = self.params["cam_trans"][0, 1, 0].item()

    def initialize_camera_pose(self, curr_time_idx, forward_prop):
        """
            Use dynamic model for the initial guess;
            add new cam frame into `self.params`
        """
        with torch.no_grad():
            if curr_time_idx > 1 and forward_prop:
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                prev_rot1 = F.normalize(self.params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
                prev_rot2 = F.normalize(self.params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
                new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
                self.params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
                # Translation
                prev_tran1 = self.params['cam_trans'][..., curr_time_idx-1].detach()
                prev_tran2 = self.params['cam_trans'][..., curr_time_idx-2].detach()
                new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                self.params['cam_trans'][..., curr_time_idx] = new_tran.detach()
            else:
                # Initialize the camera pose for the current frame using the last frame
                self.params['cam_unnorm_rots'][..., curr_time_idx] = self.params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
                self.params['cam_trans'][..., curr_time_idx] = self.params['cam_trans'][..., curr_time_idx-1].detach()

    def render_at_pose(self, c2w, white_bg=True, mask=None):
        # Get current frame Gaussians, where only the Gaussians get gradient
        # Get Frame Camera Pose
        rel_w2c = torch.linalg.inv(c2w)
        pts = self.params['means3D']
        
        # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
        pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(self.params, transformed_pts)
        depth_sil_rendervar = transformed_params2depthplussilhouette(self.params, self.first_frame_w2c,
                                                                    transformed_pts)

        # RGB Rendering
        im, radius, _, = Renderer(raster_settings=self.cam)(**rendervar)
        self.variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

        # Depth & Silhouette Rendering
        depth_sil, _, _, = Renderer(raster_settings=self.cam)(**depth_sil_rendervar)
        depth = depth_sil[0, :, :].unsqueeze(0)

        return {"render": im, "depth": depth}


    def track_rgbd(self, color, depth, gt_w2c = None, action = None):
        if not self.initialize:
            pose = torch.eye(4) if gt_w2c is None else gt_w2c
            self.init(color, depth, pose)
            return

        if self.sm.should_exit():
            self.sm.requeue()

        # Process RGB-D Data
        color = color.permute(2, 0, 1).float() / 255 # (H, W, C) -> (C, H, W)

        self.frames.append((color, depth))
        if len(self.frames) > self.win_size:
            self.frames = self.frames[-self.win_size: ]

        time_idx = self.frame_idx + 1
        gt_c2w = torch.linalg.inv(gt_w2c)
        curr_gt_w2c = gt_w2c if gt_w2c is not None else None # Place Holder here
        self.gt_w2c_all_frames.append(curr_gt_w2c)
        iter_time_idx = time_idx
        curr_data = {'cam': self.cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': self.intrinsics, 
                'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': self.gt_w2c_all_frames}

        # import pdb; pdb.set_trace()
        if time_idx > 0 and not self.config['tracking']['use_gt_poses']:
            optimizer = self.get_optimizer(tracking=True)
            self.initialize_camera_pose(time_idx, forward_prop=self.config['tracking']['forward_prop'])
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = self.params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)

            # Tracking Optimization
            iteration = 0
            do_continue_slam = False
            num_iters_tracking = self.config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(self.params, curr_data, self.variables, iter_time_idx, self.config['tracking']['loss_weights'],
                                                    self.config['tracking']['use_sil_for_loss'], self.config['tracking']['sil_thres'],
                                                    self.config['tracking']['use_l1'], self.config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                    plot_dir=self.eval_dir, visualize_tracking_loss=self.config['tracking']['visualize_tracking_loss'],
                                                    tracking_iteration=iteration)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = self.params['cam_trans'][..., time_idx].detach().clone()
                    
                    # Report Progress
                    if self.config['report_iter_progress']:
                        if self.config['use_wandb']:
                            report_progress(self.params, curr_data, iteration+1, progress_bar, iter_time_idx, sil_thres=self.config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=self.config['wandb']['save_qual'])
                        else:
                            report_progress(self.params, curr_data, iteration+1, progress_bar, iter_time_idx, sil_thres=self.config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)

                # Update the runtime numbers
                # iter_end_time = time.time()
                # tracking_iter_time_sum += iter_end_time - iter_start_time
                # tracking_iter_time_count += 1
                
                # Check if we should stop tracking
                iteration += 1
                if iteration == num_iters_tracking:
                    if losses['depth'] < self.config['tracking']['depth_loss_thres'] and self.config['tracking']['use_depth_loss_thres']:
                        break
                    elif self.config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if self.config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

        elif time_idx > 0 and self.config['tracking']['use_gt_poses']:
                # Get the ground truth pose relative to frame 0
                rel_w2c = self.gt_w2c_all_frames[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot) # shape(1,4)
                rel_w2c_tran = rel_w2c[:3, 3].detach() # shape(3,)
                # Update the camera parameters
                self.params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                self.params['cam_trans'][..., time_idx] = rel_w2c_tran

        # Mapping 
        if time_idx == 0 or (time_idx+1) % self.config['map_every'] == 0:
            # Densification
            if self.config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette & Depth Error
                self.params, self.variables = add_new_gaussians(self.config, self.params, self.variables, densify_curr_data, 
                                                      self.config['mapping']['sil_thres'], time_idx,
                                                      self.config['mean_sq_dist_method'], self.config['mapping']['densify_dict'],
                                                      add_rand_gaussians=self.config.mapping.add_rand_gaussians, downsample_pcd=self.config["downsample_pcd"])
                                                      
                post_num_pts = self.params['means3D'].shape[0]
                if self.config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = self.config['mapping_window_size'] - 2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, self.intrinsics, self.keyframe_list[:-1], num_keyframes)
                selected_time_idx = [self.keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(self.keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(self.keyframe_list[-1]['id'])
                    selected_keyframes.append(len(self.keyframe_list) - 1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = self.get_optimizer(tracking=False) 
            
            num_iters_mapping = self.config['mapping']['num_iters']
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            
            # mapping iterations
            for it in range(num_iters_mapping):
                iter_start_time = time.time()
                
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = self.keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = self.keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = self.keyframe_list[selected_rand_keyframe_idx]['depth']
                
                iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': self.cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss(self.params, iter_data, self.variables, iter_time_idx, self.config['mapping']['loss_weights'],
                                                self.config['mapping']['use_sil_for_loss'], self.config['mapping']['sil_thres'],
                                                self.config['mapping']['use_l1'], self.config['mapping']['ignore_outlier_depth_loss'], mapping=True)

                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if self.config['mapping']['prune_gaussians']:
                        self.params, self.variables = prune_gaussians(self.params, self.variables, optimizer, it, self.config['mapping']['pruning_dict'])
                        
                        if self.config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": self.params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    
                    # Gaussian-Splatting's Gradient-based Densification
                    if self.config['mapping']['use_gaussian_splatting_densification']:
                        self.params, self.variables = densify(self.params, self.variables, optimizer, it, self.config['mapping']['densify_dict'])
                        
                        if self.config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": self.params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if self.config['report_iter_progress']:
                        if self.config['use_wandb']:
                            report_progress(self.params, iter_data, it+1, progress_bar, iter_time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=self.config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(self.params, iter_data, it+1, progress_bar, iter_time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
            
            if num_iters_mapping > 0:
                progress_bar.close()

            if time_idx == 0 or (time_idx+1) % self.config['report_global_progress_every'] == 0:
                progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                with torch.no_grad():
                    if self.config['use_wandb']:
                        report_progress(self.params, curr_data, 1, progress_bar, time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=self.config['wandb']['save_qual'],
                                        mapping=True, online_time_idx=time_idx, global_logging=True)
                    else:
                        report_progress(self.params, curr_data, 1, progress_bar, time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=time_idx)
                progress_bar.close()

            self.visualize_frame(time_idx, curr_data, time_idx)

        # Add frame to keyframe list
        if ((time_idx == 0)  or  ((time_idx+1) % self.config['keyframe_every'] == 0) or \
                    (time_idx == self.config["num_frames"] - 2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                if self.config.tracking.with_droid:
                    curr_keyframe['droid_depth'] = droid_depth
                    curr_keyframe['droid_color'] = droid_color
                # Add to keyframe list
                self.keyframe_list.append(curr_keyframe)
                self.keyframe_time_indices.append(time_idx)

        # Checkpoint every iteration
        if time_idx % self.config["checkpoint_interval"] == 0:
            # compute current Uncertainty
            add_key_frame = False
            if time_idx > 0:
                H_train = None
                for keyframe in self.keyframe_list:
                    w2c = keyframe['est_w2c']

                    cur_H = self.compute_Hessian( w2c, return_points=True)
                    if H_train is None:
                        H_train = torch.zeros(*cur_H.shape, device=cur_H.device, dtype=cur_H.dtype)
                    H_train += cur_H
                    
                self.scorePoints = torch.sum(torch.reciprocal(H_train + .1), dim = 1)
                print(" non Zero Uncern: ", torch.nonzero(self.scorePoints).shape)

            self.save(time_idx)

        self.frame_idx += 1

    @torch.no_grad()
    def get_top_down_map(self, depth, c2w):
        # update current robot location on occ_map
        cam_x, cam_z = c2w[0, 3], c2w[2, 3]
        cam_pos_x = int((cam_x - self.map_center[0]) / self.cell_size + self.grid_dim[0] // 2)
        cam_pos_z = int((cam_z - self.map_center[1]) / self.cell_size + self.grid_dim[1] // 2)
        self.occ_map[2, cam_pos_z-1:cam_pos_z+2, cam_pos_x-1:cam_pos_x+2] = 1e5

        height_upper = 1.3
        height_lower = 0.1

        # # generate point cloud in current frame using points
        width, height = depth.shape[2], depth.shape[1]
        CX = self.intrinsics[0][2]
        CY = self.intrinsics[1][2]
        FX = self.intrinsics[0][0]
        FY = self.intrinsics[1][1]

        # Generate pts from depth image
        # Compute indices of pixels
        x_grid, y_grid = torch.meshgrid(torch.arange(0, width).cuda().float(), 
                                        torch.arange(0, height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY

        # sample z along depth
        sampled_z = torch.rand((11, 1, 1), device=depth.device) * 0.8 # (K, H, W)
        sampled_z.clamp_(min=0.)
        sampled_z[-1, 0, 0] = 1.              # add a point at the end of the depth range                 

        xx, yy = xx.unsqueeze(0), yy.unsqueeze(0)  # (1, H, W)
        depth_z = sampled_z * depth  # (K, H, W)
        mask = torch.bitwise_and(depth_z > 0, depth_z < 5)  # (K, H, W)

        pts = torch.stack((xx * depth_z, yy * depth_z, depth_z, torch.ones_like(depth_z)), dim = 0) # 4 x K x H x W
        free_particles = pts[:, :-1, :, :].reshape(4, -1)
        depth_pts = pts[:, -1, :, :].reshape(4, -1)

        # perform masking
        free_particles = free_particles[:, mask[:-1].reshape(-1)]
        depth_pts = depth_pts[:, mask[-1].reshape(-1)]

        # depth_pts = self.params["means3D"].permute(1, 0)

        # free particles updating
        grid = torch.empty(3, self.grid_dim[1], self.grid_dim[0], device=depth_pts.device)
        grid.fill_(0.)

        free_particles = c2w @ free_particles
        map_coords = map_utils.discretize_coords(free_particles[0], free_particles[2], self.grid_dim, self.cell_size, self.map_center)

        occ_map = torch.zeros_like(self.occ_map)

        # all particles are treated as free
        occ_lbl = torch.ones((free_particles.shape[1], 1), dtype=torch.float32, device=free_particles.device) * 2.
        
        # (N, 3) - (x, y, label)
        concatenated = torch.cat([map_coords, occ_lbl.long()], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5

        # update top-down map
        occ_map += 0.01 * grid

        # Update the top down map
        grid.fill_(0.)

        depth_pts = c2w @ depth_pts
        occ_lbl = torch.zeros((depth_pts.shape[1], 1), dtype=torch.float32, device=depth_pts.device)
        occ_sgn = torch.bitwise_and(depth_pts[1] >= height_lower, depth_pts[1] <= height_upper)
        occ_lbl[occ_sgn] = 1.
        occ_lbl[~occ_sgn] = 2.

        map_coords = map_utils.discretize_coords(depth_pts[0], depth_pts[2], self.grid_dim, self.cell_size, self.map_center)
        concatenated = torch.cat([map_coords, occ_lbl.long()], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5
        grid[1] *= 100

        # update top-down map
        occ_map += grid
        self.occ_map += occ_map / (occ_map.sum(dim=0, keepdim=True) + 1e-5)

    def build_frontiers(self):
        """ Return frontiers in pixel space  """
        prob, index = self.occ_map.max(dim=0)
        
        index = index.cpu().numpy()
        free_space = (index == 2)
        unkown = (index == 0)

        # project 
        if free_space.sum() > 18:
            lower_y, upper_y = self.cam_height - 1.0, self.cam_height
            sign = torch.bitwise_and(self.params["means3D"][:, 1] >= lower_y, self.params["means3D"][:, 1] <= upper_y)
            selected_points = self.params["means3D"][sign]
            map_coords = map_utils.discretize_coords(selected_points[:, 0], selected_points[:, 2], self.grid_dim, self.cell_size, self.map_center)
            unique_values, counts = torch.unique(map_coords, dim=0, return_counts=True)

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
        map_center = self.map_center.cpu().numpy()
        
        free_space = (labels == robot_label).astype(np.uint8)

        plt.figure()
        plt.imshow(free_space)
        plt.savefig(os.path.join(self.eval_dir, "freespace_{}.png".format(self.frame_idx)))
        plt.close()

        # perform dilation
        kernel = np.ones((3, 3), np.uint8)
        free_space_dilate = cv2.dilate(free_space.astype(np.uint8), kernel, iterations=1)
        boundary = free_space_dilate - free_space
        frontier = np.bitwise_and(boundary, unkown)
        self.frontier = frontier

        if frontier.sum() == 0:
            return None

        kernel = np.ones((3, 3), np.uint8)  
        frontier = cv2.dilate(frontier.astype(np.uint8), kernel, iterations=1) 
        
        # store frontier
        cv2.imwrite(os.path.join(self.eval_dir, "frontier_{}.png".format(self.frame_idx)), frontier.astype(np.uint8) * 255)

        # find connected components in frontier
        num_labels, labels = cv2.connectedComponents(frontier.astype(np.uint8))
        unique_label, counts = np.unique(labels, return_counts=True)
        unique_label = unique_label[1:]
        counts = counts[1:]

        # Find the largest connected component
        label_idx = np.argsort(counts)[::-1]
        select_index = min(self.selection, len(label_idx) - 1)
        largest_label = unique_label[label_idx[select_index]]
        largest_frontier = (labels == largest_label).astype(np.uint8)

        # # stuck in the same frontier
        # if self.target_fronter is not None and \
        #     (self.target_fronter * largest_frontier).sum() / self.target_fronter.sum() > 0.8:
        #     self.occ_map[2, self.target_fronter] = 0.

        #     sort_index = np.argsort(counts)[::-1]
        #     second_largest = sort_index[1]
        #     largest_frontier = (labels == second_largest).astype(np.uint8)

        # Record Frontiers
        self.target_fronter = largest_frontier

        # find the center of the largest connected component
        select_pixels = np.stack(np.where(largest_frontier), axis=1)
        # center = select_pixels.mean(axis=0)
        # center = center[[1, 0]] # switch to x,z

        # # convert to world coordinates
        # center = (center - np.array([self.grid_dim[0] // 2, self.grid_dim[1] // 2])) * self.cell_size + map_center
        
        select_pixels = select_pixels[:, [1, 0]]
        select_pixels = (select_pixels - np.array([[self.grid_dim[0] // 2, self.grid_dim[1] // 2]])) * self.cell_size + map_center[None, :]
        
        return select_pixels
    
    def generate_candidate(self, center_point):
        """ 
        sample camera poses from the center point, 
        Args:
            center_point: (K, 3) tensor, the local from which camera poses are sampled
        """

        K, radius = self.config["explore"]["sample_view_num"], self.config["explore"]["sample_range"]
        radius = min( radius * (self.selection + 1), 5 )
        theta = torch.rand((K, )).cuda() * 2 * torch.pi
        random_radius = self.config["explore"]["min_range"] + torch.rand((K, )).cuda() * (radius - self.config["explore"]["min_range"])

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

    def visualize_map(self, c2w, world_goal_point = None, path = None):
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

        if self.target_fronter is not None:
            grid_img[..., 0] = np.where(self.target_fronter, 0, grid_img[..., 0])
            grid_img[..., 1] = np.where(self.target_fronter, 255, grid_img[..., 1])
            grid_img[..., 2] = np.where(self.target_fronter, 255, grid_img[..., 2])

        if path is not None:
            for p in path:
                grid_img[p[1], p[0]] = np.array([191, 64, 191])

        plt.figure()
        plt.imshow(grid_img)
        plt.savefig(os.path.join(self.eval_dir, "occ_{}.png".format(self.frame_idx)))

    @torch.no_grad()
    def visualize_frame(self, frame_idx, curr_data, time_idx):
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(self.params, frame_idx, 
                                            gaussians_grad=False, camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(self.params, transformed_pts)
        depth_sil_rendervar = transformed_params2depthplussilhouette(self.params, curr_data["w2c"],
                                                                    transformed_pts)

        # RGB Rendering
        im, radius, _, = Renderer(raster_settings=curr_data["cam"])(**rendervar)
        
        # Depth & Silhouette Rendering
        depth_sil, _, _, = Renderer(raster_settings=curr_data["cam"])(**depth_sil_rendervar)
        depth = depth_sil[0, :, :].unsqueeze(0)
        silhouette = depth_sil[1, :, :]
        depth_sq = depth_sil[2, :, :].unsqueeze(0)
        uncertainty = depth_sq - depth**2
        uncertainty = uncertainty.detach()

        # render FishRF uncertainty map
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        mask = (curr_data['depth'] > 0) & nan_mask

        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()

        losses = calc_loss(curr_data, im, depth, mask, color_mask, 
                            True, False, False, True)

        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(im - curr_data["im"]).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(curr_data["im"].permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title("Diff RGB, Loss: {:.4f}".format(losses['im'].item()))
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title("Diff Depth, Loss: {:.4f}".format(losses['depth'].item()))
        ax[0, 3].imshow(silhouette.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(curr_data['im'].cpu().permute(1, 2, 0))
        ax[1, 3].set_title("GT Image")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Key Frame Index : {frame_idx}", fontsize=16)
        
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(self.eval_dir, exist_ok=True)

        ## Save Tracking Loss Viz
        save_plot_dir = os.path.join(self.eval_dir, "mapping".format(frame_idx))
        os.makedirs(save_plot_dir, exist_ok=True)
        plt.savefig(os.path.join(save_plot_dir, f"mapping_iter_%04d.png" % time_idx), bbox_inches='tight')
        plt.close()

    def global_planning(self, follower: habitat_sim.GreedyGeodesicFollower, agent_pose: np.array):
        """
            follower
                1. follower.pathfinder.is_navigable(point) -- whether the point is navigable
        """
        # search for fonrtiers
        frontier = self.build_frontiers()
        self.generate_Gaussian_at_frontier()
        use_frontier = frontier is not None and self.selection < 2

        # select upper 40% of the points based on H_train
        agent_pos = agent_pose[:3, 3]
        H_train = None
        for keyframe in self.keyframe_list:
            w2c = keyframe['est_w2c']

            cur_H = self.compute_Hessian( w2c, return_points=True, random_gaussian_params=False)
            if H_train is None:
                H_train = torch.zeros(*cur_H.shape, device=cur_H.device, dtype=cur_H.dtype)
            H_train += cur_H
        
        H_train_inv = torch.reciprocal(H_train + 0.1)
        scorePoints = torch.sum(H_train_inv, dim = 1) # (N, )
        # assert scorePoints.shape[0] == self.params["means3D"].shape[0]

        with torch.no_grad():
            # Random generate camera position pointing to center point
            K = self.config["explore"]["sample_view_num"]

            if use_frontier:
                frontier_height = np.ones((frontier.shape[0], )) * self.cam_height
                frontier = np.stack([frontier[:, 0], frontier_height, frontier[:, 1]], axis=1)
                frontier = torch.from_numpy(frontier).cuda()

                frontier_rand_index = torch.randint(0, frontier.shape[0], (K, ))
                center_point = frontier[frontier_rand_index]

                selected_points_index = None

            # no frontier, no rand gaussians, nothing changes
            else:
                # In habitat, y is upward
                height_range = self.config["explore"]["height_range"]
                lower_y, upper_y = self.cam_height - height_range, self.cam_height + height_range

                sign = torch.bitwise_and(self.params["means3D"][:, 1] >= lower_y, self.params["means3D"][:, 1] <= upper_y)
                selected_points_xyz = self.params["means3D"][sign]
                selected_scores = scorePoints[sign]
                points_index_range = torch.where(sign)[0]

                threshold = torch.quantile(selected_scores, 0.8)
                selected_points_xyz_thresholded = selected_points_xyz[selected_scores > threshold]

                # select next target
                if len(selected_points_xyz_thresholded) > 0:
                    points_index = points_index_range[selected_scores > threshold]
                    selected_scores_np = selected_scores[selected_scores > threshold].cpu().numpy()

                    selected_points_np = selected_points_xyz_thresholded.cpu().numpy()
                    # choose the maximum point
                    # center_point = selected_points_np[np.argmax(selected_scores_np)]

                    # Cluster selected_points_xyz points using DBSCANn
                    clustering = DBSCAN(eps=0.1, min_samples=5)
                    clustering.fit(selected_points_np)
                    labels = clustering.labels_

                    # Get the cluster with the largest number of points
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    # max_count_label = unique_labels[np.argmax(counts)]

                    max_count_label = -1
                    max_score = -1
                    for label_cls in unique_labels:
                        # -1 is noise from DBSCAN
                        if label_cls  < 0:
                            continue

                        cluster_score = selected_scores_np[labels == label_cls].max()
                        if  cluster_score > max_score:
                            max_count_label = label_cls
                            max_score = cluster_score

                    segmentated_labels = np.ones((len(scorePoints), )) * -1
                    segmentated_labels[points_index.cpu().numpy()] = labels
                    points_index_range_np = points_index_range.cpu().numpy()
                    segmentated_labels_range = segmentated_labels[points_index_range_np]
                    np.savez(os.path.join(self.eval_dir, f"global_planning_iter{self.frame_idx}.npz"), 
                                segmentated_labels = segmentated_labels_range, max_label = max_count_label, points_index_range = points_index_range_np)
                    
                    selected_points_np = selected_points_np[labels == max_count_label]
                    selected_points_index = points_index[labels == max_count_label]
                    
                    selected_points_np_rand_index = np.random.randint(0, selected_points_np.shape[0], (K, ))
                    center_point = selected_points_np[selected_points_np_rand_index]

                    center_point = torch.from_numpy(center_point).cuda()
                else:
                    center_point = selected_points_xyz[torch.argmax(selected_scores)]
                    center_point = center_point.unsqueeze(0)
                    selected_points_index = None

            # sample camera poses
            c2ws = self.generate_candidate(center_point)

            scores = []
            navigable_c2w = []
            action_lens = []
            
            max_points_score = torch.zeros((scorePoints.shape[0], )).cuda()
            for cam_id, c2w in enumerate(tqdm(c2ws, desc="Examing Hessains")):
                target_pos = c2w[:3, 3].cpu().numpy()
                target_pos[1] = agent_pos[1]
                if follower.pathfinder.is_navigable(target_pos):
                    try:
                        follower.reset()
                        action_lens.append(len(list(follower.find_path(target_pos))))
                    except Exception as e:
                        # Some time the follower throws an exception even if it's navigable
                        logger.warn(f"Exception found when finding action lens for {target_pos}")
                        # print(e)
                        continue
                    w2c = torch.linalg.inv(c2w)
                    cur_H = self.compute_Hessian( w2c, return_points=True, random_gaussian_params=False)
                    
                    # update max point scores
                    pointScores = torch.sum(cur_H * H_train_inv, dim=1)
                    max_points_score = torch.where(max_points_score > pointScores, max_points_score, pointScores)   

                    # only visualize the second round
                    # if self.selection == 2:
                    #     points = self.params["means3D"].cpu().numpy()
                    #     cur_H_score = pointScores.cpu().numpy()
                    #     extrinsic = w2c.cpu().numpy()
                    #     np.savez(os.path.join(self.eval_dir, f"pcd_curH_select_{self.selection}_cam_{cam_id}.npz"), 
                    #              points=points, score=cur_H_score, extrinsic=extrinsic)

                    view_score = torch.sum(cur_H * H_train_inv).item()
                    scores.append(view_score)
                    navigable_c2w.append(c2w)
            
            # culling invisible
            if self.config["explore"]["prune_invisible"] and selected_points_index is not None:
                selected_points_max_score = max_points_score[selected_points_index]
                filter_index = torch.where(selected_points_max_score < scorePoints[selected_points_index] * 2)[0]
                gaussian_index = selected_points_index[filter_index]
                remove_index = torch.zeros((self.params["means3D"].shape[0], ), dtype=torch.bool).cuda()
                remove_index[gaussian_index] = True
                print(f"Remove {len(gaussian_index)} gaussians due to low H")
                self.params, self.variables = remove_points(remove_index, self.params, self.variables, None)
                torch.cuda.empty_cache()

            self.selection += 1
            if len(navigable_c2w) == 0:
                # import pdb; pdb.set_trace()
                return None, None

            scores = torch.tensor(scores)
            navigable_c2w = torch.stack(navigable_c2w)

            return scores, navigable_c2w
        
    def compute_H_train(self, random_gaussians=None):
        H_train = None
        for keyframe in self.keyframe_list:
            w2c = keyframe['est_w2c']

            cur_H = self.compute_Hessian( w2c, return_points=True, random_gaussian_params=False)
            if H_train is None:
                H_train = torch.zeros(*cur_H.shape, device=cur_H.device, dtype=cur_H.dtype)
            H_train += cur_H

        return H_train 
    
    def gs_pts_cnt(self, random_gaussian_params=None):
        """ API Setting """
        return 1
       
    def pose_eval(self, poses, random_gaussian_params=None):
        """ Compute pose scores for the poses """
        H_train = self.compute_H_train()
        H_train_inv = torch.reciprocal(H_train + 0.1)

        scores = []
        navigable_c2ws = []

        for cam_id, c2w in enumerate(tqdm(poses, desc="Examing Hessains")):
            # compute cur H
            w2c = torch.linalg.inv(c2w)
            cur_H = self.compute_Hessian( w2c, return_points=True, random_gaussian_params=False)

            self.render_at_pose(c2w, random_gaussian_params)
            
            view_score = torch.sum(cur_H * H_train_inv).item() 
            
            scores.append(view_score)
            navigable_c2ws.append(c2w)
        
        scores = torch.tensor(scores)
        navigable_c2w = torch.stack(navigable_c2ws)
        
        return scores, navigable_c2w
        
    def delete_gaussians_by_index(self, gaussian_index):
        num_gaussians = self.params["means3D"].shape[0]
        keep_index = torch.ones((num_gaussians, ), dtype=torch.bool).cuda()
        keep_index[gaussian_index] = False

        for k, v in self.params.items():
            if k in ["means3D", "rgb_colros", "unnorm_rotations", "log_scales"]:
                self.params[k] = v[keep_index]

        for k, v in self.variables.items():
            if k in ['max_2D_radius', 'means2D_gradient_accum', 'denom', 'timestep', 'means2D', 'seen']:
                self.variables[k] = v[keep_index]

    def DFS_acq_score_planning(self, train_poses, pathfinder):
        max_depth = 6
        current_pose = train_poses[-1]

        # compute H_train
        H_train = None

        # for train_pose in train_poses:
        for keyframe in self.keyframe_list:
            w2c = keyframe['est_w2c']

            cur_H = self.compute_Hessian(w2c)
            if H_train is None:
                H_train = torch.zeros(cur_H.shape[0], device=cur_H.device, dtype=cur_H.dtype)
            H_train += cur_H

        def DFS(train_H,  next_pos, action_id, depth):
            if depth > 0:
                if pathfinder.is_navigable( next_pos[:3, 3] ):
                    cur_H = self.compute_Hessian( np.linalg.inv(next_pos))
                    acq_score = (cur_H * torch.reciprocal(H_train + .1)).sum().item()
                    next_train_H = train_H + cur_H
                else:
                    # for un Navigable points, just return
                    return -1, []
            else:
                # For the root node, nothing to do
                acq_score = 0.
                next_train_H = train_H

            if depth == max_depth:
                return acq_score, []
            else:
                # compute forward score
                forward_pos = compute_next_campos(next_pos.copy(), 1)
                f_score, f_action = DFS(next_train_H.clone(), forward_pos, 1, depth + 1)

                # compute left score
                if action_id != 3:
                    left_pos = compute_next_campos(next_pos.copy(), 2)
                    l_score, l_action = DFS(next_train_H.clone(), left_pos, 2, depth + 1)
                else:
                    l_score, l_action = -1, []

                # compute right score
                if action_id != 2:
                    right_pos = compute_next_campos(next_pos.copy(), 3)
                    r_score, r_action = DFS(next_train_H.clone(), right_pos, 3, depth + 1)
                else:
                    r_score, r_action = -1, []

                scores = np.array([f_score, l_score, r_score])
                actions = [f_action, l_action, r_action]

                best_action_id = np.argmax(scores)
                best_action = best_action_id + 1
                best_actions = actions[best_action_id]
                best_actions.append(best_action)

                return acq_score + scores[best_action_id], best_actions

        acq_score, action_list = DFS(cur_H, current_pose, 1, 0)
        return action_list

    def save(self, time_idx):
        save_params_ckpt(self.params, self.eval_dir, time_idx)
        np.save(os.path.join(self.eval_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(self.keyframe_time_indices))
        
    def get_optimizer(self, tracking):
        # get lr rate
        if tracking:
            lrs_dict = self.config["tracking"]["lrs"]
        else:
            lrs_dict = self.config["mapping"]["lrs"]

        param_groups = [{'params': [v], 'name': k, 'lr': lrs_dict[k]} for k, v in self.params.items()]
        if tracking:
            return torch.optim.Adam(param_groups)
        else:
            return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        
    def generate_Gaussian_at_frontier(self):
        self.frontier_gaussian = {}

        if self.frontier.sum() > 0:
            pos_z, pos_x = np.nonzero(self.frontier)
            pos = np.stack([pos_x, pos_z], axis=1)
            pos_w = self.convert_to_world(pos) # (NUM_GRID, 2)

            position = torch.from_numpy(pos_w).cuda().float()

            # add gaussians
            GAUSSIAN_PER_GRID = 100
            xz_offset = torch.rand((1, GAUSSIAN_PER_GRID, 2)).cuda() * self.cell_size
            y_offset = (self.cam_height - 1.0) + torch.rand((position.shape[0], GAUSSIAN_PER_GRID, 1), device=position.device)

            new_p3t = torch.cat([position[:, None, :] + xz_offset, y_offset], dim=-1)
            new_p3t = new_p3t.reshape(-1, 3)

            new_rgb_colors = torch.rand((new_p3t.shape[0], 3)).cuda()
            new_rotations = torch.zeros((new_p3t.shape[0], 4)).cuda()
            new_rotations[:, 0] = 1.
            new_opacities = torch.ones((new_p3t.shape[0], 1)).cuda()
            new_scales = torch.ones((new_p3t.shape[0], 3)).cuda() * self.cell_size

            self.frontier_gaussian.update({
                "means3D": new_p3t,
                "rgb_colors": new_rgb_colors,
                "unnorm_rotations": new_rotations,
                "logit_opacities": new_opacities,
                "log_scales": new_scales
            })
        
    @torch.enable_grad()
    def compute_Hessian(self, rel_w2c, return_points = False, 
                        random_gaussian_params = False, 
                        return_pose = False):
        """
            Compute uncertainty at candidate pose
                params: Gaussian slam params
                candidate_trans: (3, )
                candidate_rot: (4, )
                return_points:
                    if True, then the Hessian matrix is returned in shape (N, C), 
                    else, it is flatten in 1-D.

        """
        if isinstance(rel_w2c, np.ndarray):
            rel_w2c = torch.from_numpy(rel_w2c).cuda()
        rel_w2c = rel_w2c.float()

        # transform to candidate frame
        with torch.no_grad():
            pts = self.params['means3D']
            pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
            pts4 = torch.cat((pts, pts_ones), dim=1)

            transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
            rgb_colors = self.params['rgb_colors']
            rotations = F.normalize(self.params['unnorm_rotations'])
            opacities = torch.sigmoid(self.params['logit_opacities'])
            scales = torch.exp(self.params['log_scales'])
            if scales.shape[-1] == 1: # isotropic
                scales = torch.tile(scales, (1, 3))

        num_points = transformed_pts.shape[0]

        rendervar = {
            'means3D': transformed_pts.requires_grad_(True),
            'colors_precomp': rgb_colors.requires_grad_(True),
            'rotations': rotations.requires_grad_(True),
            'opacities': opacities.requires_grad_(True),
            'scales': scales.requires_grad_(True),
            'means2D': torch.zeros_like(transformed_pts, requires_grad=True, device="cuda") + 0
        }
        params_keys = ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"]

        # for means3D, rotation won't change sum of square since R^T R = I
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=self.cam, backward_power=2)(**rendervar)
        im.backward(gradient=torch.ones_like(im) * 1e-3)

        if return_points:
            cur_H = torch.cat([transformed_pts.grad.detach().reshape(num_points, -1),  
                                opacities.grad.detach().reshape(num_points, -1)], dim=1)

        else:
            cur_H = torch.cat([transformed_pts.grad.detach().reshape(-1), 
                                opacities.grad.detach().reshape(-1)])
            
        # set grad to zero
        for k, v in rendervar.items():
            v.grad.fill_(0.)

        if not return_pose:
            return cur_H
        else:
            return cur_H, torch.eye(6).cuda()
    
    def get_latest_frame(self):
        """
        
        Return:
            (4, 4) c2w transform
        """
        # get the latest c2w transform
        curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., self.frame_idx].detach())
        curr_cam_tran = self.params['cam_trans'][..., self.frame_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        
        curr_c2w = torch.linalg.inv(curr_w2c)
        curr_c2w = curr_c2w.cpu().numpy()
        return curr_c2w
    
    @property
    def cur_frame_idx(self):
        return self.frame_idx
    
    def get_gaussian_xyz(self):
        return self.params['means3D']
    
    @property
    def gaussian_points(self):
        return self.get_gaussian_xyz()
    
    def pause(self):
        """ API to be compatible with Mono GS """
        return

    def resume(self):
        """ API to be compatible with Mono GS """
        return
    
    def color_refinement(self):
        """ API to be compatible with Mono GS """
        return
    
    def stop(self):
        """ API to be compatible with Mono GS """
        return
