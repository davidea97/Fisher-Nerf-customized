from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import queue
import cv2
import glob
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm
from planning.astar import AstarPlanner, LocalizationError
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat.utils.visualizations import maps
from datasets.dataloader import HabitatDataScene
# from models.gaussian_slam import GaussianSLAM, PruneException
from models.utils import PruneException, ssim
from models.SLAM.gaussian import GaussianSLAM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import datasets.util.utils as utils
import os
from copy import deepcopy
import imageio
import pickle
import time
from tqdm import tqdm
import math
import json as js
import sys

from scipy.spatial.transform import Rotation as SciR
from models.SLAM.utils.slam_external import calc_psnr
# from models.gaussian_slam.gaussian_splatting.utils.loss_utils import ssim
from models.SLAM.utils.slam_external import compute_next_campos, build_rotation
from cluster_manager import ClusterStateManager
from models.UPEN import UPEN
from habitat.core.simulator import AgentState

from configs.base_config import get_cfg_defaults
import datasets.util.utils as utils
from test_utils import draw_map, set_agent_state

import shutil
import logging
from rich.logging import RichHandler
import wandb
from einops import rearrange, reduce, repeat

from visualization.habitat_viz import HabitatVisualizer
from IPython import embed

import open3d as o3d

# Frontier policy exploration
from frontier_exploration.frontier_search import FrontierSearch
from frontier_exploration.map import *

# FORMAT = "%(pathname)s:%(lineno)d %(message)s"
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")
cm = ClusterStateManager()

# Flip Z-Y axis from habitat OpenGL convention to common convention
habitat_transform = np.array([
                        [1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., -1., 0.],
                        [0., 0., 0., 1.]
                    ])

np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)

def pos_quant2w2c(pos: np.ndarray, quat: np.ndarray, agent_state: AgentState): 
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

    # render at position 
    c2w = utils.get_cam_transform(agent_state=agent_state) @ habitat_transform
    c2w_t = torch.from_numpy(c2w).float().cuda()
    w2c = torch.linalg.inv(c2w_t)
    return w2c

def save_pointcloud(rgb, depth, intrinsics, pose, save_path, obj_mask=None):
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    if obj_mask is not None:
        assert obj_mask.shape == depth.shape, "Object mask must have same shape as depth"

        # Invalidate depth where the object is masked
        depth = depth.copy()
        depth[obj_mask > 0] = 0.0  # set depth to 0 (ignored by Open3D)

        # Optionally, also mask the RGB to avoid color noise (not necessary, but can help)
        rgb = rgb.copy()
        rgb[obj_mask > 0] = 0

    # Create Open3D RGBD image
    rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))  # depth in meters

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=100.0)

    # Intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    # Point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Apply camera-to-world transformation
    if pose is not None:
        pcd.transform(pose)

    # Save
    o3d.io.write_point_cloud(save_path, pcd)


def count_visible_points(global_pts, c2w, intrinsics, img_size):
    """
    Return the 3D points visible from the given camera pose (ignoring depth bounds).
    
    Args:
        global_pts (np.ndarray): (N, 3) point cloud in world coordinates.
        c2w (np.ndarray): (4, 4) camera-to-world matrix.
        intrinsics (tuple): (fx, fy, cx, cy)
        img_size (tuple): (H, W)
    
    Returns:
        visible_pts (np.ndarray): subset of global_pts visible from camera
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    H, W = img_size, img_size
    
    # Transform points to camera frame
    w2c = np.linalg.inv(c2w)
    pts_cam = (w2c[:3, :3] @ global_pts.T + w2c[:3, 3:4]).T  # (N, 3)

    # Filter points in front of camera
    valid_z = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid_z]
    visible_world_pts = global_pts[valid_z]

    # Project to image plane
    u = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
    v = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy

    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    visible_pts = visible_world_pts[in_frame]

    return visible_pts

class NoFrontierError(Exception):
    pass

class Navigator(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options, scene_id, dynamic_scene, dino_extraction, save_data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options

        # Load config
        self.slam_config = get_cfg_defaults()
        self.slam_config.merge_from_file(options.slam_config)
        self.dynamic_scene = dynamic_scene
        self.save_data = save_data
        self.dino_extraction = dino_extraction
        os.makedirs(os.path.join(self.slam_config["workdir"], self.slam_config["run_name"]), exist_ok=True)
        write_config_file = os.path.join(self.slam_config["workdir"], self.slam_config["run_name"], "config.yaml")
        if os.path.exists(write_config_file):
            # if file already exists, then reload.
            logger.info(f"loading existing config at {write_config_file}")
            self.slam_config.merge_from_file(write_config_file)
        else:
            # copy config file
            shutil.copy(self.options.slam_config, write_config_file)
        
        if self.options.max_steps != self.slam_config["num_frames"]:
            logger.warn(f"max_steps {self.options.max_steps} != self.slam_config['num_frames'] {self.slam_config['num_frames']}, override self.options")
            self.options.max_steps = self.slam_config["num_frames"]
        
        self.options.img_size = self.slam_config.img_height
        assert self.slam_config.img_height == self.slam_config.img_width, "Only square images are supported for now"

        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        # build summary dir
        summary_dir = os.path.join(self.options.log_dir, scene_id)
        summary_dir = os.path.join(summary_dir, 'tensorboard')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # tensorboardX SummaryWriter for use in save_summaries
        self.summary_writer = SummaryWriter(summary_dir)

        self.scene_id = scene_id
        if self.options.split=="val":
            if self.options.noisy_actions:
                config_file = self.options.config_val_file_noisy
            else:
                config_file = self.options.config_val_file
        elif self.options.split=="test":
            if self.options.noisy_actions:
                config_file = self.options.config_test_file_noisy
            else:
                config_file = self.options.config_test_file

        # Load config
        self.slam_config = get_cfg_defaults()
        self.slam_config.merge_from_file(self.options.slam_config)
        # If we wish to overwrite the run_name, we need to do it beforei createing the dir
        # Don't use multi layer directoryu because wandb doesn't support it
        self.slam_config["run_name"] = f"{self.scene_id}-{self.slam_config.run_name}"

        os.makedirs(os.path.join(self.slam_config["workdir"], self.slam_config["run_name"]), exist_ok=True)
        write_config_file = os.path.join(self.slam_config["workdir"], self.slam_config["run_name"], "config.yaml")
        if os.path.exists(write_config_file):
            # if file already exists, then reload.
            logger.info(f"loading existing config at {write_config_file}")
            self.slam_config.merge_from_file(write_config_file)
            self.slam_config["run_name"] = f"{self.scene_id}-{self.slam_config.run_name}"
        else:
            # copy config file
            shutil.copy(self.options.slam_config, write_config_file)

        if self.options.max_steps != self.slam_config["num_frames"]:
            logger.warn(f"max_steps {self.options.max_steps} != self.slam_config['num_frames'] {self.slam_config['num_frames']}, override self.options")
            self.options.max_steps = self.slam_config["num_frames"]

        self.slam_config["policy"]["workdir"] = self.slam_config["workdir"]
        self.slam_config["policy"]["run_name"] = self.slam_config["run_name"]
        self.slam_config.freeze()

        # Get the current time
        current_time = datetime.datetime.now()
        # Format the time as month-day-hour-minute
        formatted_time = current_time.strftime("%m-%d-%H-%M")
        wandb_id = "{}-{}".format(self.slam_config["run_name"], formatted_time)

        wandb.init(project="active_mapping", id=wandb_id, config=self.slam_config, resume='allow',
                    mode=None if self.slam_config.use_wandb else "disabled",
                )

        self.options.max_steps = self.slam_config["num_frames"]
        self.options.forward_step_size = self.slam_config["forward_step_size"]
        self.options.turn_angle = self.slam_config["turn_angle"]
        self.options.occupancy_height_thresh = self.slam_config["policy"]["occupancy_height_thresh"]
    
        self.test_ds = HabitatDataScene(self.options, config_file=config_file, slam_config=self.slam_config, scene_id=self.scene_id, dynamic=dynamic_scene)
        self.step_count = 0
        self.min_depth, self.max_depth = self.test_ds.min_depth, self.test_ds.max_depth

        self.policy_name = self.slam_config["policy"]["name"]
        # print("Policy name: ", self.policy_name)
        # agent_state = self.test_ds.sim.sim.get_agent_state()

        if self.policy_name in ["DFS", "global_local_plan", "oracle", "pose-comp"]:
            self.policy = None
        elif self.policy_name in ["astar_greedy", "frontier", "only_frontier"]:
            # Exploration parameters:
            self.policy = AstarPlanner(
                self.slam_config, os.path.join(self.slam_config["workdir"], self.slam_config["run_name"])
            )
        elif self.policy_name == "UPEN":
            self.policy = UPEN(self.options, self.slam_config["policy"])
        elif self.policy_name == "TrajReader":
            action_seq_file = f"{self.scene_id}.txt"
            self.traj_poses = np.loadtxt(action_seq_file, delimiter=',')

            print("[WARN] Set the max steps to {} ".format(self.traj_poses.shape[0]))
            self.options.max_steps = self.traj_poses.shape[0]
        else:
            assert False, f"unkown policy name {self.slam_config['policy']['name']}"

        self.policy_eval_dir = os.path.join(self.slam_config["workdir"], self.slam_config["run_name"])
        self.habvis = HabitatVisualizer(self.policy_eval_dir, scene_id) 
        self.cfg = self.slam_config # unified abberavation

        # Initialize a 3D global pointcloud that we want to fill
        self.global_pcd = o3d.geometry.PointCloud()

    def store_pointcloud(self, rgb, depth, intrinsics, pose):
        height, width = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Create Open3D RGBD image
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))  # depth in meters

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
            depth_trunc=100.0)

        # Intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

        # Point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # Apply camera-to-world transformation
        if pose is not None:
            pcd.transform(pose)

        # Add to global point cloud
        self.global_pcd += pcd

    def store_filtered_pointcloud(self, rgb, depth, intrinsics, pose, keep_ratio=0.05, step = None):
        height, width = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Create Open3D RGBD image
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))  # depth in meters

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
            depth_trunc=100.0)

        # Intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

        # Point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # Transform to world
        if pose is not None:
            pcd.transform(pose)
        
        # print("Point cloud size before filtering: ", len(pcd.points))
        # Remove NaN or infinite points
        pcd.remove_non_finite_points()
        # print("Point cloud size after filtering: ", len(pcd.points))
        # Convert to numpy
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Skip if empty
        if len(pts) == 0:
            return

        # Subsample points: keep only N% of them
        num_points = pts.shape[0]
        num_to_keep = int(keep_ratio * height * width)

        if num_points > num_to_keep:
            indices = np.random.choice(num_points, size=num_to_keep, replace=False)
            pts = pts[indices]
            colors = colors[indices]
        
        # Rebuild the point cloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(pts)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors)
        # print("Filtered point cloud size: ", len(filtered_pcd.points))

        pts_tensor = torch.from_numpy(pts).float()         # shape: (N, 3)
        colors_tensor = torch.from_numpy(colors).float()   # shape: (N, 3)

        if not hasattr(self, 'global_pts_tensor'):
            self.global_pts_tensor = pts_tensor
            self.global_colors_tensor = colors_tensor
        else:
            self.global_pts_tensor = torch.vstack([self.global_pts_tensor, pts_tensor])
            self.global_colors_tensor = torch.vstack([self.global_colors_tensor, colors_tensor])
        
        # print("Global point cloud size: ", len(self.global_pts_tensor))

        # # Add to global map
        # self.global_pcd += filtered_pcd
        # print("Global point cloud size after adding: ", len(self.global_pcd.points))

        # if step ==20:
        #     save_path = os.path.join(self.policy_eval_dir, f"pointcloud/global_pcl_{step}.ply")
        #     o3d.io.write_point_cloud(save_path, self.global_pcd)

    def add_dynamic_object(self):

        from SimObjects import SimObject

        obj_templates_mgr = self.test_ds.sim._sim.get_object_template_manager()
        rigid_obj_mgr = self.test_ds.sim._sim.get_rigid_object_manager()
        # template_file_path = os.path.join(self.options.root_path, "habitat_example_objects_0.2/car")
        template_file_path = os.path.join(self.options.root_path, "habitat_example_objects_0.2/space_robot")
        scale_factor = 0.1

        template_id = obj_templates_mgr.load_configs(
            str(template_file_path))[0]
        obj_template = obj_templates_mgr.get_template_by_id(template_id)
        obj_template.scale = [scale_factor, scale_factor, scale_factor]

        obj_templates_mgr.register_template(obj_template)
        # create sphere
        new_obj = rigid_obj_mgr.add_object_by_template_id(template_id)
        # self.default_agent = self.sim._sim.get_agent(0)
        # self.rigid_obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC   # It moves the object from the initial position (it cannot fly because the dynamic is enabled and it falls down)
        agent_node = self.test_ds.sim._sim.agents[0].scene_node
        # Place object 1.0 meter in front of the camera
        camera_forward_offset = [0.0, 1.0, -1.0]  # -Z is forward in Habitat-Sim
        object_position = agent_node.transformation.transform_point(camera_forward_offset)
        new_obj.translation = object_position
        move_object = True
        show_object_axes = False
        sim_obj = SimObject(new_obj, moving=move_object, show_object_axes=show_object_axes)
        if move_object:
            linear_velocity = [0.0, 0.0, 3.0]
            angular_velocity = [0.0, -5.0, 0.0]
            sim_obj.enable_kinematic_velocity(linear_velocity, angular_velocity)
        
        return new_obj

    @torch.no_grad()
    def frontier_test_navigation(self):
        agent_state = self.test_ds.sim.sim.get_agent_state()
        rgb_pos = agent_state.sensor_states['rgb'].position
        self.test_ds.sim.sim.reset()
        agent_state = self.test_ds.sim.sim.get_agent_state()
        rgb_pos = agent_state.sensor_states['rgb'].position
        episode = None        
        observations_cpu = self.test_ds.sim.sim.get_sensor_observations()
        observations = {"rgb": torch.from_numpy(observations_cpu["rgb"]).cuda(), "depth": torch.from_numpy(observations_cpu["depth"]).cuda(), "semantic": torch.from_numpy(observations_cpu["depth"]).cuda()}
        img = observations['rgb'][:, :, :3]
        depth_init = observations['depth'].reshape(1, observations['depth'].shape[0], observations['depth'].shape[1])
        # semantic = observations['semantic'].reshape(1, self.test_ds.img_size[0], self.test_ds.img_size[1])
        
        if self.dynamic_scene:
            new_obj = self.add_dynamic_object()

        # Get Camera to World transform
        c2w = utils.get_cam_transform(agent_state=self.test_ds.sim.sim.get_agent_state()) @ habitat_transform
        c2w_t = torch.from_numpy(c2w).float().cuda()
        w2c_t = torch.linalg.inv(c2w_t)
                
        # resume SLAM system if neededs
        slam = GaussianSLAM(self.slam_config)
        # slam.init(img, depth_init, w2c_t)

        # resume from slam
        t = slam.cur_frame_idx + 1

        if slam.cur_frame_idx > 0:
            c2w = slam.get_latest_frame()
            set_agent_state(self.test_ds.sim.sim, c2w)

        # reset agent from TrajReader
        if self.policy_name == "TrajReader":
            set_agent_state(self.test_ds.sim.sim, np.concatenate([self.traj_poses[t, :3], self.traj_poses[t, 3:]]))

        observations = self.test_ds.sim.sim.get_sensor_observations()
        observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}
        img = observations['rgb'][:, :, :3] # (H, W, 3)
        # depth = observations['depth'].reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1) # (H, W, 1)
        init_c2w = utils.get_cam_transform(agent_state=self.test_ds.sim.sim.get_agent_state()) @ habitat_transform

        intrinsics = torch.linalg.inv(self.test_ds.inv_K).cuda()
        print("Intrinsics: ", intrinsics)
        self.abs_poses = []

        # init local policy, in the first steps it explores the local area by scanning the environment
        action_queue = self.init_local_policy(slam, init_c2w, intrinsics, episode)

        agent_episode_distance = 0.0 # distance covered by agent at any given time in the episode
        previous_pos = self.test_ds.sim.sim.get_agent_state().position

        planned_path = None
        goal_pose = None
        action_id = -1
        expansion = 1

        if self.save_data:
            os.makedirs(os.path.join(self.policy_eval_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "depth"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "semantic"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "bw_mask"), exist_ok=True)
        os.makedirs(os.path.join(self.policy_eval_dir, "pointcloud"), exist_ok=True)

        # === Load DINOv2 ===   
        if self.dynamic_scene:
            from scripts.dino_extract import DINOExtract, extract_dino_features, dino_image_visualization

            dino_export="../third_party/dino_models/dinov2_vitl14_pretrain.pth"
            dino_extractor = DINOExtract(dino_export, feature_layer=1)
            print("DINOv2 model loaded")

        print("Max steps: ", self.options.max_steps)
        try: 
            all_dino_descriptors = []
            all_images = []
            all_selected_coord = []
            while t < self.options.max_steps:

                print(f"##### UPDATING STEP: {t} #####")
                start_step_time = time.time()
                img = observations['rgb'][:, :, :3]
                # depth = observations['depth'].reshape(1, self.test_ds.img_size[0], self.test_ds.img_size[1])
                depth = observations['depth'].reshape(1, observations['depth'].shape[0], observations['depth'].shape[1])
                agent_state = self.test_ds.sim._sim.get_agent_state()
                agent_rotation = agent_state.rotation  # quaternion: x, y, z, w
                agent_translation = agent_state.position

                quat = [agent_rotation.w, agent_rotation.x, agent_rotation.y, agent_rotation.z]  # w, x, y, z
                rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quat)

                pose = utils.get_cam_transform(agent_state=self.test_ds.sim.sim.get_agent_state()) @ habitat_transform

                # Save single rgb, depth, semantic for debugging
                observations_cpu = self.test_ds.sim.sim.get_sensor_observations()
                rgb_bgr = cv2.cvtColor(observations_cpu["rgb"], cv2.COLOR_RGB2BGR)
                depth_vis_gray = (observations_cpu["depth"] / 10.0 * 255).astype(np.uint8)
                depth_raw = (observations_cpu["depth"])
                depth_vis = cv2.cvtColor(depth_vis_gray, cv2.COLOR_GRAY2BGR)
                semantic_obs_uint8 = (observations_cpu["semantic"] % 40).astype(np.uint8)
                semantic_vis = d3_40_colors_rgb[semantic_obs_uint8]
                if self.dynamic_scene:
                    
                    object_mask = (observations_cpu["semantic"] == new_obj.semantic_id).astype(np.uint8) * 255
                    object_mask_bw = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
                    if np.array(object_mask_bw).ndim == 3:

                        # Use only the first channel, assuming grayscale video auto-expanded to RGB
                        object_mask_bw = object_mask_bw[:, :, 0]
                    if self.dino_extraction:
                        dino_descriptors, selected_coord = extract_dino_features(rgb_bgr, object_mask_bw, dino_extractor)
                        if dino_descriptors.shape[0] == 0:
                            print("DINO descriptors are empty!")
                        else:
                            all_dino_descriptors.append(dino_descriptors)
                            all_images.append(rgb_bgr)
                            all_selected_coord.append(selected_coord)
                            print("Dino descriptors shape: ", dino_descriptors.shape)

                save_path = os.path.join(self.policy_eval_dir, f"pointcloud/pcl_{t}.ply")
                if self.save_data:
                    cv2.imwrite(os.path.join(self.policy_eval_dir, f"rgb/rgb_{t}.png"), rgb_bgr)
                    cv2.imwrite(os.path.join(self.policy_eval_dir, f"depth/depth_{t}.png"), depth_vis)
                    cv2.imwrite(os.path.join(self.policy_eval_dir, f"semantic/semantic_{t}.png"), semantic_vis)
                    cv2.imwrite(os.path.join(self.policy_eval_dir, f"bw_mask/bw_{t}.png"), object_mask_bw)                    
                    save_pointcloud(rgb_bgr, depth_raw, intrinsics, pose, save_path, None)

                # Store pcl to the global pcl
                # self.store_pointcloud(rgb_bgr, depth_raw, intrinsics, pose)
                
                self.store_filtered_pointcloud(rgb_bgr, depth_raw, intrinsics, pose, keep_ratio=0.05, step=t)
                
                global_pts = self.global_pts_tensor.cpu().numpy()
                np_intrinsics = intrinsics.cpu().numpy()
                start_time_occ = time.time()
                # points_visible = count_visible_points(global_pts, pose, np_intrinsics, self.options.img_size)
                
                c2w = utils.get_cam_transform(agent_state=self.test_ds.sim.sim.get_agent_state()) @ habitat_transform
                # print("Camera 2 world: ", c2w)
                c2w_t = torch.from_numpy(c2w).float().cuda()
                w2c_t = torch.linalg.inv(c2w_t)

                if cm.should_exit():
                    cm.requeue()
                
                # 3d info
                
                agent_pose, agent_height = utils.get_sim_location(agent_state=self.test_ds.sim.sim.get_agent_state())
                self.abs_poses.append(agent_pose)

                # Update habitat vis tool and save the current state
                # if self.save_data:
                if t % 2 == 0:
                    self.habvis.save_vis_seen(self.test_ds.sim.sim, t)
                self.habvis.update_fow_sim(self.test_ds.sim.sim)
                
                # save habvis
                if (slam.cur_frame_idx) % self.slam_config["checkpoint_interval"] == 0 and slam.cur_frame_idx > 0:
                    save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    self.habvis.save(save_path)

                if self.policy_name == "frontier":
                    
                    if (slam.cur_frame_idx) % self.slam_config["checkpoint_interval"] == 0 and slam.cur_frame_idx > 0 :
                        save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                        self.policy.save(save_path)
                    agent_state = self.test_ds.sim._sim.get_agent_state()
                    agent_rotation = agent_state.rotation  # quaternion: x, y, z, w
                    agent_translation = agent_state.position

                    quat = [agent_rotation.w, agent_rotation.x, agent_rotation.y, agent_rotation.z]  # w, x, y, z
                    rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
                    current_agent_pose = np.eye(4)
                    current_agent_pose[:3, :3] = rot_matrix
                    current_agent_pose[:3, 3] = agent_translation

                    # current_agent_pose = slam.get_latest_frame()
                    current_agent_pos = current_agent_pose[:3, 3]
                    
                    # update occlusion map
                    
                    self.policy.update_occ_map(depth, c2w_t, t, self.slam_config["downsample_pcd"])
                    print("Occlusion map update time: ", time.time() - start_time_occ)
                    print("End step time: ", time.time() - start_step_time)
                    # self.policy.visualize_map(c2w)
                    # print("Action queue size: ", action_queue.qsize())
                    while action_queue.empty():
                        # pause backend during evaluation
                        # slam.pause()
                        
                        best_path = None
                        while best_path is None:
                            current_agent_pos = current_agent_pose[:3, 3]
                            
                            # global plan -- select global 
                            global_points, _, _ = \
                                    self.policy.global_planning_frontier(expansion, visualize=True, 
                                                                agent_pose=current_agent_pos)

                            if global_points is None:
                                raise NoFrontierError("No frontier found")

                            global_points = global_points.cpu().numpy()
                            
                 
                            # plan actions for each global goal
                            _, path_actions, paths_arr = self.action_planning_frontier(global_points, current_agent_pose, t)
                            best_path = path_actions[0]
                            map_path = paths_arr[0]


                        if best_path is None:
                            print("No best path! Turning")
                            expansion += 1
                            if not action_queue.full():
                                action_queue.put(2)
                        else:
                            expansion = 1
                            # Fill into action queue
                            print(best_path)
                            for action_id in best_path:
                                if not action_queue.full():
                                    action_queue.put(action_id)
                                else:
                                    break
                    
                        # resume backend process after planning
                        # slam.resume()

                        # visualize map
                        self.policy.visualize_map(c2w, goal_pose, map_path)
                    
                    action_id = action_queue.get()
                    # time.sleep(1.)

                # explicitly clear observation otherwise they will be kept in memory the whole time
                observations = None
                
                # Apply next action
                # depth is [0, 1] (should be rescaled to 10)
                prev_pos = self.test_ds.sim.sim.get_agent_state().position
                self.test_ds.sim.sim.step(action_id)
                current_pos = self.test_ds.sim.sim.get_agent_state().position
                # function to check if the agent is stuck
                if isinstance(self.policy, AstarPlanner) and action_id == 1 \
                    and np.max(np.abs(prev_pos - current_pos)) < 1e-3:
                    # robot stuck
                    print("robot stuck, replan")
                    current_agent_pose = slam.get_latest_frame()
                    current_agent_pos = current_agent_pose[:3, 3]

                    head_theta = math.atan2(current_agent_pose[0, 2], current_agent_pose[2, 2])
                    start = self.policy.convert_to_map(current_agent_pos[[0, 2]])[[1, 0]] # from x-z to z-x
                    
                    # set obstacle according to collision
                    if -np.pi/4 <= head_theta and head_theta <= np.pi/4:
                        self.policy.occ_map[1, start[0] + 3, start[1]] = 1000
                    elif np.pi/4 <= head_theta and head_theta <= 3 * np.pi/4:
                        self.policy.occ_map[1, start[0], start[1] + 3] = 1000
                    elif -3 * np.pi/4 <= head_theta and head_theta <= - np.pi/4:
                        self.policy.occ_map[1, start[0], start[1] - 3] = 1000
                    else:
                        self.policy.occ_map[1, start[0] - 3, start[1]] = 1000

                    logger.warn(" cannot move, clear action queue, replan! ")
                    while not action_queue.empty():
                        action_id = action_queue.get()
                
                # get new observation
                observations = self.test_ds.sim.sim.get_sensor_observations()
                observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}

                # if slam.config.Training.pose_filter:
                #     slam.update_motion_est(action_id, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])

                # estimate distance covered by agent
                agent_episode_distance += utils.euclidean_distance(current_pos, previous_pos)
                previous_pos = current_pos
                t += 1

                # if self.cfg.eval_every > 0 and (t + 1) % self.cfg.eval_every == 0:
                #     print("Evaluating at step: ", t)
                #     self.eval_navigation(slam, t)

        except NoFrontierError as e:
            pass
        except LocalizationError as e:
            logger.error("Robot inside obstacle")
            pass

        print("Agent episode distance: ", agent_episode_distance)
        # slam.color_refinement()
        # self.eval_navigation(slam, t)

        if self.slam_config.use_wandb:
            wandb.finish()
        
        # Close current scene
        self.test_ds.sim.sim.close()
        # slam.frontend.backend_queue.put(["stop"])
        slam.stop()



    def log(self, output, log_step=0):
        for k in output:
            self.summary_writer.add_scalar(k, output[k], log_step)

        if self.slam_config.use_wandb:
            wandb.log(output, self.step_count)

    def action_planning_frontier(self, global_points, current_agent_pose, t):
        """
        Plan sequences of actions for each goal poses

        Args:
            global_points (np.array): (N, 4, 4) goal poses
            current_agent_pose (np.array): (4, 4) current agent pose
            t: time step

        Return:
            valid_global_points (List[np.array]): valid goal poses
            path_actions: List[List[int]]: action for each goal
            paths_arr: List[] List for each planned path
        """
        print(">> A* planning for each goal pose")
        valid_global_points = []
        path_actions = []
        paths_arr = []

        # set start position in A* Planner
        current_agent_pos = current_agent_pose[:3, 3]
        start = self.policy.convert_to_map(current_agent_pos[[0, 2]])[[1, 0]] # from x-z to z-x
        self.policy.setup_start(start, None, t)

        for pose_np in tqdm(global_points, desc="Action Planning"):
            
            if cm.should_exit():
                cm.requeue()

            pos_np = pose_np[:3, 3].copy()
            pos_np[1] = current_agent_pos[1] # set the same heights

            finish = self.policy.convert_to_map(pos_np[[0, 2]])[[1, 0]] # convert to z-x
            paths = self.policy.planning(finish) # A* Planning in [x, z]

            if len(paths) == 0:
                continue
            
            future_pose = current_agent_pose.copy()

            # set to cam height
            future_pose[1, 3] = self.policy.cam_height
            stage_goal_idx = 1

            # if only rotation
            if len(paths) == 1:
                paths = np.concatenate([paths, finish[None, :]], axis=0)

            # current_pos = np.array(paths[0])[[1, 0]] # from z-x to x-z
            stage_goal = paths[stage_goal_idx]
            
            stage_goal_w = self.policy.convert_to_world(stage_goal + 0.5) # grid cell center
            stage_goal_w = np.array([stage_goal_w[0], future_pose[1, 3], stage_goal_w[1], 1])
            path_action = []
            generate_opposite_turn = False

            # push actions to queue
            while len(path_action) < self.slam_config["policy"]["planning_queue_size"]:
                # compute action
                rel_pos = np.linalg.inv(future_pose) @ stage_goal_w
                xz_rel_pos = rel_pos[[0, 2]]
                
                if np.linalg.norm(xz_rel_pos) < self.slam_config["forward_step_size"]:
                    stage_goal_idx += 1 
                    if stage_goal_idx == len(paths):
                        # change orientation

                        angle = np.rad2deg(math.atan2(pose_np[0, 2], pose_np[2, 2])) - \
                                np.rad2deg(math.atan2(future_pose[0, 2], future_pose[2, 2]))

                        if abs(angle) > 180:
                            angle = angle - 360 if angle > 0 else angle + 360

                        num_actions = int(abs(angle) // self.slam_config["turn_angle"])
                        for k in range(num_actions):
                            if len(path_action) < self.slam_config["policy"]["planning_queue_size"]:
                                
                                action = 2 if angle > 0 else 3
                                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])

                                # append action
                                path_action.append(action)

                            else:
                                break
                        
                        # break 
                        break
                    
                    else:
                        # move to next stage goal
                        stage_goal = paths[stage_goal_idx]
                        stage_goal_w = self.policy.convert_to_world(stage_goal)
                        stage_goal_w = np.array([stage_goal_w[0], future_pose[1, 3], stage_goal_w[1], 1])
                        rel_pos = np.linalg.inv(future_pose) @ stage_goal_w
                        xz_rel_pos = rel_pos[[0, 2]]
                    
                angle = np.arctan2(xz_rel_pos[0], xz_rel_pos[1])

                # generate turn in the opposite direction for each path if possible 
                # if abs(angle) > np.pi / 2 and not generate_opposite_turn:
                #     generate_opposite_turn = True
                #     action = 3 if angle < 0 else 2 # opposite turn
                #     opposite_action = path_action + [action] * (self.slam_config["policy"]["planning_queue_size"] - len(path_action))
                    
                #     # put into the final results
                #     path_actions.append(opposite_action)
                #     valid_global_points.append(pose_np)
                #     paths_arr.append(paths)

                if angle > np.radians(self.slam_config["turn_angle"]):
                    action = 3
                elif angle < - np.radians(self.slam_config["turn_angle"]):
                    action = 2
                else:
                    action = 1
                    
                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])
                path_action.append(action)

            if path_action not in path_actions:
                path_actions.append(path_action)
                valid_global_points.append(pose_np)
                paths_arr.append(paths)
        
        return valid_global_points, path_actions, paths_arr

    def init_local_policy(self, slam,
                        init_c2w,
                        intrinsics,
                        episode = None) -> queue.Queue:
        """
        Init the local policy

        slam: the Gaussian SLAM system
        init_c2w: the initial camera pose
        intrinsics: the camera intrinsics
        episode: episode information, set to None
        """
        action_queue = queue.Queue(maxsize=100)
        
        if self.policy_name in ["astar_greedy", "frontier"]:
            self.policy.init(init_c2w, intrinsics)
            print("Init Astar Policy: ", self.policy_name)
            if slam.cur_frame_idx > 0:
                folder = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                ckpt_path = os.path.join(folder, "astar.pth")
                print("Loading Astar Policy from: ", ckpt_path)
                self.policy.load(ckpt_path)
                t = slam.cur_frame_idx
            else:
                # turn around for initialization
                init_scan_steps = 72 if not self.options.debug else 2
                # for k in range(2):
                for k in range(init_scan_steps):
                    action_queue.put(2)
        
        elif self.policy_name == "UPEN":
            self.policy.init(self.test_ds, episode)

            if slam.cur_frame_idx > 0:
                slam.pause()
                
                # load trajectory
                traj_path = os.path.join(folder, "traj.npz")
                traj = np.load(traj_path)["est_traj"]

                for curr_w2c in tqdm(traj, desc="Loading Traj ..."):
                    curr_c2w = np.linalg.inv(curr_w2c)
                    set_agent_state(self.test_ds.sim.sim, curr_c2w)

                    observations = None
                    observations = self.test_ds.sim.sim.get_sensor_observations()
                    observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}
                    depth = observations['depth'].reshape(1, self.test_ds.img_size[0], self.test_ds.img_size[1])

                    agent_pose, y_height = utils.get_sim_location(agent_state=self.test_ds.sim.sim.get_agent_state())
                    self.abs_poses.append(agent_pose)
                    self.policy.predict_action(t, self.abs_poses, depth) 
                    t += 1

                slam.resume()

        # reset habvis
        self.habvis.reset()
        habvis_size = 768 if not hasattr(self, "policy") else self.policy.grid_dim[0]
        self.habvis.set_map(self.test_ds.sim.sim, habvis_size)

        # load from checkpoint
        if slam.cur_frame_idx > 0:
            folder = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
            self.habvis.load(folder)
        
        return action_queue
