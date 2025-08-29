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
from models.SLAM.gaussian_object import GaussianObjectSLAM
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
from scripts.dino_extract import DINOExtract, extract_dino_features, dino_image_visualization
from SimObjects import SimObject
from scipy.spatial.transform import Rotation as SciR
from models.SLAM.utils.slam_external import calc_psnr
# from models.gaussian_slam.gaussian_splatting.utils.loss_utils import ssim
from models.SLAM.utils.slam_external import compute_next_campos, build_rotation
from cluster_manager import ClusterStateManager
from models.UPEN import UPEN
from habitat.core.simulator import AgentState

from configs.base_config import get_cfg_defaults
import datasets.util.utils as utils
from test_utils import draw_map, set_agent_state, check_camera_pose_wrt_map, novelty_mask_from_pcd_nn, check_camera_pose_wrt_map_with_mesh

import shutil
import logging
from rich.logging import RichHandler
import wandb
from einops import rearrange, reduce, repeat

from visualization.habitat_viz import HabitatVisualizer
from IPython import embed

import open3d as o3d
import magnum as mn

# Frontier policy exploration
from frontier_exploration.frontier_search import FrontierSearch
from frontier_exploration.map import *

import trimesh
import pyrender
from scripts.evaluation import load_glb_pointcloud, load_ply_pointcloud, apply_transform_to_pointcloud, get_latest_pcl_file, save_pointcloud_as_ply, load_env_glb_pointcloud, concat_mesh_from_glb
from scripts.eval_3d_reconstruction import accuracy_comp_ratio_from_pcl

from utils.object_reconstruction_utils import mask_border_contact, estimate_object_center, object_center_error, object_size_ratio, reached
from utils.dino_utils import DinoBank, to_numpy

from test_utils import yaml_safe_load, yaml_safe_dump

# FORMAT = "%(pathname)s:%(lineno)d %(message)s"
OBJ_SCALE_FACTOR = 0.5
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

rotation_90_x = np.array([
                        [1., 0.,  0., 0.],
                        [0., 0., -1., 0.],
                        [0., 1.,  0., 0.],
                        [0., 0.,  0., 1.]
                    ])

rotation_m90_x = np.array([
                        [1., 0.,  0., 0.],
                        [0., 0., 1., 0],
                        [0., -1.,  0., 0],
                        [0., 0.,  0., 1.]
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

def create_video_from_images(img_dir, output_path, fps=10):
    from natsort import natsorted
    image_paths = natsorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    if not image_paths:
        print("No images found in", img_dir)
        return
    
    # Read the first image to get dimensions
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape
    
    # Define video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        out.write(frame)
    
    out.release()
    print("Video saved to", output_path)

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

class NoFrontierError(Exception):
    pass

class NavTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options, scene_id, object_scene, dynamic_scene, dynamic_scene_rec, dino_extraction, save_data, save_map, known_env=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options

        # Load config
        self.slam_config = get_cfg_defaults()
        self.slam_config.merge_from_file(options.slam_config)

        self.object_scene = object_scene
        self.dynamic_scene = dynamic_scene
        self.dynamic_scene_rec = dynamic_scene_rec
        self.dino_extraction = dino_extraction
        self.save_data = save_data
        self.save_map = save_map
        self.object_tracking = False
        self.init_object_slam = False
        self.init_object_slam_done = False
        self.action_queue = None  # Queue for actions to be executed by the policy

        # self.gaussian_optimization = gaussian_optimization

        if self.options.max_steps != self.slam_config["num_frames"]:
            logger.warn(f"max_steps {self.options.max_steps} != self.slam_config['num_frames'] {self.slam_config['num_frames']}, override self.options")
            self.options.max_steps = self.slam_config["num_frames"]
        
        self.options.img_size = self.slam_config.img_height
        assert self.slam_config.img_height == self.slam_config.img_width, "Only square images are supported for now"

        # for k in self.options.__dict__.keys():
        #     print(k, self.options.__dict__[k])

        # build summary dir
        summary_dir = os.path.join(self.options.log_dir, scene_id)
        summary_dir = os.path.join(summary_dir, 'tensorboard')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # tensorboardX SummaryWriter for use in save_summaries
        self.summary_writer = SummaryWriter(summary_dir)
        self.scene_id = scene_id

        # Navigation config file for Habitat
        nav_config_file = self.options.config_val_file

        # Load config
        # self.slam_config = get_cfg_defaults()
        # self.slam_config.merge_from_file(self.options.slam_config)
        # If we wish to overwrite the run_name, we need to do it beforei createing the dir
        # Don't use multi layer directory because wandb doesn't support it
        self.slam_config["run_name"] = f"{self.scene_id}-{self.slam_config.run_name}"

        # Create the directory for the experiment results
        os.makedirs(os.path.join(self.slam_config["workdir"], self.slam_config["run_name"], self.options.dataset_type), exist_ok=True)

        # Save the run config file
        write_config_file = os.path.join(self.slam_config["workdir"], self.slam_config["run_name"], self.options.dataset_type, "config.yaml")
        shutil.copy(self.options.slam_config, write_config_file)

        # if self.options.max_steps != self.slam_config["num_frames"]:
        #     logger.warn(f"max_steps {self.options.max_steps} != self.slam_config['num_frames'] {self.slam_config['num_frames']}, override self.options")
        #     self.options.max_steps = self.slam_config["num_frames"]

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

        self.habitat_ds = HabitatDataScene(self.options, config_file=nav_config_file, slam_config=self.slam_config, scene_id=self.scene_id, dynamic=dynamic_scene)

        self.step_count = 0
        self.min_depth, self.max_depth = self.habitat_ds.min_depth, self.habitat_ds.max_depth
        self.policy_name = self.slam_config["policy"]["name"]
        print(">> Policy name: ", self.policy_name)

        if self.policy_name in ["DFS", "global_local_plan", "oracle", "pose-comp"]:
            self.policy = None
        elif self.policy_name in ["gaussians_based", "frontier", "random_walk"]:
            # Exploration parameters:
            self.policy = AstarPlanner(
                self.slam_config, os.path.join(self.slam_config["workdir"], self.slam_config["run_name"], self.options.dataset_type)
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

        self.policy_eval_dir = os.path.join(self.slam_config["workdir"], self.slam_config["run_name"], self.options.dataset_type)
        self.habvis = HabitatVisualizer(self.policy_eval_dir, scene_id) 
        self.cfg = self.slam_config # unified abberavation

        # Initialize a 3D global pointcloud that we want to fill
        self.global_pcd = o3d.geometry.PointCloud()
        self.global_obj_pcd = o3d.geometry.PointCloud()

        if known_env is not None:
            self.known_env_mode = True
            self.known_env_path = known_env
            print("Known env path: ", self.known_env_path)
            self.pcd_known_env = load_env_glb_pointcloud(self.known_env_path, apply_transform=habitat_transform)
            self.gt_3d_oriented_w = apply_transform_to_pointcloud(self.pcd_known_env, rotation_90_x)
            # save_pointcloud_as_ply(self.gt_3d_oriented_w, "gt_scene.ply")
            self.tau_m = 0.2
        else:
            self.known_env_mode = False
            self.known_env_path = None
            self.pcd_known_env = None
            self.gt_3d_oriented_w = None


        # Dynamic Object initialization
        if self.object_scene:
            # self.camera_forward_offset = [-2.0, 0.0, -1.0]
            # self.camera_forward_offset = [0.0, 1.5, -1.0]
            self.camera_forward_offset = [1.3, 1.5, 2.0]
            self.dynamic_object_path = "habitat_example_objects_0.2/wheeled_robot"
            # find the glb file within the dynamic_object_path
            self.obj_glb_files = glob.glob(os.path.join(self.options.root_path, self.dynamic_object_path, "*.glb"))
            print("Object GLB files found:", self.obj_glb_files)

            self.obj_pts = load_glb_pointcloud(self.obj_glb_files[0])
            rotation_m90_x = np.array([
                [1., 0.,  0., 0.],
                [0., 0., 1., 0],
                [0., -1.,  0., 0],
                [0., 0.,  0., 1.]
            ])

            self.gt_obj_3d_rotated = apply_transform_to_pointcloud(self.obj_pts, rotation_m90_x, scale=OBJ_SCALE_FACTOR)
            # save_pointcloud_as_ply(self.gt_obj_3d_rotated, "gt_object.ply")
            self.obj_pcd = o3d.geometry.PointCloud()
            self.obj_pcd.points = o3d.utility.Vector3dVector(self.gt_obj_3d_rotated)
            
            # self.dynamic_mesh = trimesh.load("path/to/object.glb")
            self.sim_obj = self.initialize_dynamic_object(self.dynamic_object_path, self.camera_forward_offset)
    
    def initialize_dynamic_object(self, path_obj, camera_forward_offset, scale_factor=OBJ_SCALE_FACTOR):
        obj_templates_mgr = self.habitat_ds.sim._sim.get_object_template_manager()
        rigid_obj_mgr = self.habitat_ds.sim._sim.get_rigid_object_manager()
        template_file_path = os.path.join(self.options.root_path, path_obj)

        template_id = obj_templates_mgr.load_configs(
            str(template_file_path))[0]
        # print("Template ID:", template_id)
        obj_template = obj_templates_mgr.get_template_by_id(template_id)
        obj_template.scale = [scale_factor, scale_factor, scale_factor]
        obj_templates_mgr.register_template(obj_template)
        
        # Add object
        new_obj = rigid_obj_mgr.add_object_by_template_id(template_id)
        new_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        # Fixed object starting pose
        agent_node = self.habitat_ds.sim._sim.agents[0].scene_node
        # object_position = agent_node.transformation.transform_point(camera_forward_offset)
        # new_obj.translation = object_position

        # === RANDOMLY PLACE OBJECT ===
        pathfinder = self.habitat_ds.sim._sim.pathfinder
        found = False
        max_tries = 100
        for _ in range(max_tries):
            sample_pos = pathfinder.get_random_navigable_point()
            if sample_pos is not None:
                found = True
                break
        if not found:
            raise RuntimeError("Couldn't find a random navigable point for the object.")
        
        print("Agent position:", agent_node.transformation.translation)
        print("Sample position for dynamic object:", sample_pos)
        new_obj.translation = sample_pos

        # Optionally, random orientation (around Y-axis)
        # yaw = np.random.uniform(0, 2 * np.pi)
        # new_obj.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())

        sim_obj = SimObject(new_obj, dynamic=self.dynamic_scene)
        return sim_obj

    def backproj_depth_to_pcl(self, rgb, depth, intrinsics, pose, keep_ratio=0.05, step = None):

        # if isinstance(depth, torch.Tensor) and depth.dim() == 3 and depth.shape[0] == 1:
        #     depth = depth.squeeze(0) 

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

        return filtered_pcd, pts, colors

    def project_points_to_image(self, pcd, intrinsics, T_cam_world, T_obj_world, image_shape):
        # Trasforma la PCD nel sistema di riferimento camera
        T_world_cam = np.linalg.inv(T_cam_world)
        T_obj_cam = T_world_cam @ T_obj_world
        pcd.transform(T_obj_cam)  # inplace

        # Ottieni punti in camera space
        points = np.asarray(pcd.points)
        points = points[points[:, 2] > 0]  # keep only in front of camera
        intrinsics = intrinsics.cpu().numpy() if torch.is_tensor(intrinsics) else intrinsics
        intrinsics = intrinsics[:3, :3]  # Ensure intrinsics is 3x3
        # Proiezione
        points_2d = (intrinsics @ points.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]

        h, w = image_shape
        u = np.round(points_2d[:, 0]).astype(int)
        v = np.round(points_2d[:, 1]).astype(int)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[v[valid], u[valid]] = 255

        return mask

    def store_filtered_pointcloud(self, rgb, depth, intrinsics, pose, keep_ratio=1, step = None):
        # print("Intrinsics: ", intrinsics)
        filtered_pcd, pts, colors = self.backproj_depth_to_pcl(rgb, depth, intrinsics, pose, keep_ratio, step)

        pts_tensor = torch.from_numpy(pts).float()         # shape: (N, 3)
        colors_tensor = torch.from_numpy(colors).float()   # shape: (N, 3)

        if not hasattr(self, 'global_pts_tensor'):
            self.global_pts_tensor = pts_tensor
            self.global_colors_tensor = colors_tensor
        else:
            self.global_pts_tensor = torch.vstack([self.global_pts_tensor, pts_tensor])
            self.global_colors_tensor = torch.vstack([self.global_colors_tensor, colors_tensor])
        
        # print("Global point cloud size: ", len(self.global_pts_tensor))

        # Add to global map
        self.global_pcd += filtered_pcd
        # print("Global point cloud size after adding: ", len(self.global_pcd.points))

        if step == self.options.max_steps -1 or (step + 1) % 100 == 0:
            save_path = os.path.join(self.policy_eval_dir, f"pointcloud/global_pcl_{step}.ply")
            o3d.io.write_point_cloud(save_path, self.global_pcd)

    def store_filtered_obj_pointcloud(self, rgb, depth, intrinsics, object_mask_bw, camera_pose, object_pose, step=None):
    
        # height, width = depth.shape
        intrinsics = intrinsics.cpu().numpy() if hasattr(intrinsics, 'cpu') else intrinsics

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        valid_mask = (object_mask_bw > 0) & (depth > 0)
        ys, xs = np.where(valid_mask)
        if len(ys) == 0:
            return

        zs = depth[ys, xs]
        xs_cam = (xs - cx) * zs / fx
        ys_cam = (ys - cy) * zs / fy
        points_cam = np.stack([xs_cam, ys_cam, zs], axis=1)

        points_hom = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)  # (N, 4)
        points_world = (camera_pose @ points_hom.T).T[:, :3]

        T_world_object = np.linalg.inv(object_pose)
        points_obj = (T_world_object @ np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1).T).T[:, :3]

        colors = rgb[ys, xs].astype(np.float32) / 255.0

        filtered_points = points_obj
        filtered_colors = colors

        # Skip if empty
        if len(filtered_points) == 0:
            return
        
        # Rebuild the point cloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Add to global map
        self.global_obj_pcd += filtered_pcd
        save_path = os.path.join(self.policy_eval_dir, f"pointcloud/global_pcl_obj_{step}.ply")
        o3d.io.write_point_cloud(save_path, self.global_obj_pcd)

        return filtered_pcd, filtered_points, points_world


    @torch.no_grad()
    def test_navigation(self):
        self.habitat_ds.sim.sim.reset()
        agent_node = self.habitat_ds.sim._sim.agents[0].scene_node

        if self.object_scene:
            initial_obj_pos = agent_node.transformation.transform_point(self.camera_forward_offset)
            pathfinder = self.habitat_ds.sim._sim.pathfinder
            object_position = None

            object_position = initial_obj_pos
            # Snap the object position to the nearest navigable point
            ground_pos = mn.Vector3(*pathfinder.snap_point(object_position))

            # Get bounding box height

            object_height = 0.5
            # Offset to place base of object just above ground
            offset_y = object_height / 2.0 + 0.01  # + small epsilon to avoid z-fighting
            ground_pos.y += offset_y

            self.sim_obj.set_translation(ground_pos)

        episode = None
        observations_cpu = self.habitat_ds.sim.sim.get_sensor_observations()
        observations = {"rgb": torch.from_numpy(observations_cpu["rgb"]).cuda(), "depth": torch.from_numpy(observations_cpu["depth"]).cuda(), "semantic": torch.from_numpy(observations_cpu["semantic"]).cuda()}
        img = observations['rgb'][:, :, :3]
        depth = observations['depth'].reshape(1, self.habitat_ds.img_size[0], self.habitat_ds.img_size[1])

        c2w = utils.get_cam_transform(agent_state=self.habitat_ds.sim.sim.get_agent_state()) @ habitat_transform
        c2w_t = torch.from_numpy(c2w).float().cuda()
        w2c_t = torch.linalg.inv(c2w_t)

        # resume SLAM system if neededs
        slam = GaussianSLAM(self.slam_config)
        slam.init(img, depth, c2w_t)

        # resume from slam
        # DAVIDE
        t = slam.cur_frame_idx + 1
        t = slam.cur_frame_idx

        if slam.cur_frame_idx > 0:
            c2w = slam.get_latest_frame()
            set_agent_state(self.habitat_ds.sim.sim, c2w)

        # reset agent from TrajReader
        if self.policy_name == "TrajReader":
            set_agent_state(self.habitat_ds.sim.sim, np.concatenate([self.traj_poses[t, :3], self.traj_poses[t, 3:]]))
        init_c2w = utils.get_cam_transform(agent_state=self.habitat_ds.sim.sim.get_agent_state()) @ habitat_transform

        intrinsics = torch.linalg.inv(self.habitat_ds.inv_K).cuda()
        # print("Intrinsics: ", intrinsics)
        self.abs_agent_poses = []

        # init local policy
        self.init_local_policy(slam, init_c2w, intrinsics, episode, known_env_mode=self.known_env_mode)

        agent_episode_distance = 0.0 # distance covered by agent at any given time in the episode
        previous_pos = self.habitat_ds.sim.sim.get_agent_state().position

        planned_path = None
        goal_pose = None
        action_id = -1
        expansion = 1
        if self.save_data:
            os.makedirs(os.path.join(self.policy_eval_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "rgb_mask"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "depth"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "depth_vis"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "semantic"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "bw_mask"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "pointcloud"), exist_ok=True)
            os.makedirs(os.path.join(self.policy_eval_dir, "camera_poses"), exist_ok=True)

         # === Load DINOv2 ===   
        if self.dino_extraction:
            from scripts.dino_extract import DINOExtract, extract_dino_features, dino_image_visualization
            self.dino_bank = DinoBank(
                max_frames=20,              # come volevi
                max_points_per_frame=2000,  # evita memoria eccessiva
                tau_pool_add=0.7,
                tau_cham_add=0.7,
                tau_match_add=0.50,
                seed=0
            )
            self.last_map_frame = -10**9
            self.COOLDOWN = 5
            dino_export="../third_party/dino_models/dinov2_vitl14_pretrain.pth"
            dino_extractor = DINOExtract(dino_export, feature_layer=1)
            print("DINOv2 model loaded")
        
        print("Max steps: ", self.options.max_steps)
        try: 
            all_dino_descriptors = []
            all_images = []
            all_selected_coord = []
            saved_camera_poses = []
            robot_stuck_count = 0
            while t < self.options.max_steps:
                print(f"##### NAVIGATION STEP: {t} #####")
                img = observations['rgb'][:, :, :3]
                depth = observations['depth'].reshape(1, self.habitat_ds.img_size[0], self.habitat_ds.img_size[1])
                               
                if self.object_scene:
                    dt = 0.2
                    current_pos = np.array(self.sim_obj.get_translation())
                    local_velocity = mn.Vector3(self.sim_obj.get_linear_velocity())
                    rotation = self.sim_obj.obj.rotation
                    global_velocity = rotation.transform_vector(local_velocity)

                    next_pose = current_pos + dt*global_velocity

                    is_valid = self.habitat_ds.sim.sim.pathfinder.is_navigable(next_pose)

                    # self.sim_obj.moving_forward_and_back(is_valid)
                    self.sim_obj.moving_randomly(is_valid)
                    object_pose = self.sim_obj.get_transformation() @ habitat_transform
                    object_pose = np.array(object_pose)
                   
                # c2w = utils.get_cam_transform(agent_state=self.habitat_ds.sim.sim.get_agent_state()) @ habitat_transform
                # c2w_t = torch.from_numpy(c2w).float().cuda()
                # w2c_t = torch.linalg.inv(c2w_t)
                observations_cpu = self.habitat_ds.sim.sim.get_sensor_observations()
                rgb_bgr = cv2.cvtColor(observations_cpu["rgb"], cv2.COLOR_RGB2BGR)
                depth_vis_gray = (observations_cpu["depth"] / 10.0 * 255).astype(np.uint8)
                depth_raw = (observations_cpu["depth"])

                depth_vis = cv2.cvtColor(depth_vis_gray, cv2.COLOR_GRAY2BGR)
                semantic_obs_uint8 = (observations_cpu["semantic"] % 40).astype(np.uint8)
                semantic_vis = d3_40_colors_rgb[semantic_obs_uint8]

                camera_pose = utils.get_cam_transform(agent_state=self.habitat_ds.sim.sim.get_agent_state()) @ habitat_transform
                if self.known_env_mode == True:
                    c2w_t = torch.from_numpy(camera_pose).float().cuda()
                    inv_K_t = self.habitat_ds.inv_K.cuda().float()
                    H, W = self.habitat_ds.img_size
                    mask = novelty_mask_from_pcd_nn(self.gt_3d_oriented_w, depth_raw, inv_K_t, c2w_t, (H, W), z_forward="Z", dist_thresh_m=0.05, stride=1, frame_id=t)

                    # self.policy.cover_fov_2d(c2w_t, fov_deg=90.0, max_range=4.0, ang_step_deg=2.0)
                    # free_cells = (self.policy.occ_map[2] > 0).sum().item()
                    # cov_cells  = self.policy.covered.sum().item()
                    # if free_cells > 0 and (cov_cells / free_cells) >= 0.95:
                    #     print(f">> Coverage complete: {(cov_cells/free_cells)*100:.1f}%")
                    #     break

                    # fronts = self.policy.build_frontier_cells()
                    # if not fronts:
                    #     print(">> No frontier left.")
                    #     break

                    # print("Frontier cells: ", len(fronts))
                    # print("Fronts: ", fronts)

                    # self.policy.visualize_occ_map(slam.get_latest_frame()[:3, 3])

                    
                # Store all the camera poses
                saved_camera_poses.append(camera_pose.copy())
                if self.save_data:
                    cv2.imwrite(os.path.join(self.policy_eval_dir, f"rgb/rgb_{t}.png"), rgb_bgr)
                # Detect object in the scene
                if self.dynamic_scene_rec:
                    image_shape = rgb_bgr.shape[:2]
                    # mask = self.project_points_to_image(self.obj_pcd, intrinsics, c2w, object_pose, image_shape)
                    # print(f"Mask {t} found object points: {np.sum(mask > 0)}")
                    object_mask = (observations_cpu["semantic"] == self.sim_obj.get_semantic_id()).astype(np.uint8) * 255

                    if self.known_env_mode == False:
                        object_mask_bw = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
                    else:
                        mask = mask.astype(np.uint8)*255
                        object_mask_bw = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                    if np.array(object_mask_bw).ndim == 3:
                        object_mask_bw = object_mask_bw[:, :, 0]
                    
                    # Save the object mask if and only if it is not empty
                    if np.any(object_mask_bw!=0):
                        cv2.imwrite(os.path.join(self.policy_eval_dir, f"bw_mask/bw_{t}.png"), object_mask_bw)
                        mask_bool = object_mask_bw != 0
                        masked_rgb = np.zeros_like(rgb_bgr)
                        masked_rgb[mask_bool] = rgb_bgr[mask_bool]
                        cv2.imwrite(os.path.join(self.policy_eval_dir, f"rgb_mask/masked_rgb_{t}.png"), masked_rgb)

                        rgb_bgr_converted = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                        self.store_filtered_obj_pointcloud(rgb_bgr_converted, depth_raw, intrinsics, object_mask_bw, camera_pose, object_pose, step=t)
                        self.evaluate_3d_object_reconstruction(step=t)
                        do_mapping_this_frame=True
                        if self.dino_extraction:
                            dino_descriptors, selected_coord = extract_dino_features(rgb_bgr, object_mask_bw, dino_extractor)
                            if dino_descriptors.shape[0] == 0:
                                print("DINO descriptors are empty!")
                            else:
                                D_t = to_numpy(dino_descriptors).astype(np.float32, copy=False)

                                # metriche di similarità contro TUTTI i keyframe salvati
                                sim_pool_max, sim_ch, frac_fwd, frac_bwd = self.dino_bank.similarity_metrics(D_t)
                                print(f"[DINO] maxPool={sim_pool_max:.3f}  chamfer={sim_ch:.3f}  "
                                    f"fwd>0.8={frac_fwd:.2f}  bwd>0.8={frac_bwd:.2f}  bank={len(self.dino_bank)}")

                                # decisione mapping (percettiva) – puoi combinarla con baseline/parallasse/EIG
                                very_similar = (sim_pool_max > 0.95) and (sim_ch > 0.82) and (min(frac_fwd, frac_bwd) > 0.60)
                                cooldown_ok = (t - self.last_map_frame) >= self.COOLDOWN
                                do_mapping_this_frame = (not very_similar) and cooldown_ok

                                if do_mapping_this_frame:
                                    # ... la tua fase di mapping ...
                                    self.last_map_frame = t

                                # aggiorna il bank SOLO se la vista è *distintiva* (come volevi)
                                added = self.dino_bank.add_if_distinct(D_t, force=False)
                                all_dino_descriptors.append(dino_descriptors)
                                all_images.append(rgb_bgr)
                                all_selected_coord.append(selected_coord)
                                print("Dino descriptors shape: ", dino_descriptors.shape)

                # Save current rgb, depth, semantic, camera pose
                if self.save_data:
                    # save_path = os.path.join(self.policy_eval_dir, f"pointcloud/pcl_{t}.ply")
                    
                    cv2.imwrite(os.path.join(self.policy_eval_dir, f"depth_vis/depth_{t}.png"), depth_vis)
                    cv2.imwrite(os.path.join(self.policy_eval_dir, f"semantic/semantic_{t}.png"), semantic_vis)
                    # np.save(os.path.join(self.policy_eval_dir, f"depth/depth_map_{t}.npy"), depth_raw)
                    np.save(os.path.join(self.policy_eval_dir, f"camera_poses/pose_{t}.npy"), c2w)
                    depth_mm = (depth_raw * 1000).astype(np.uint16)
                    cv2.imwrite(os.path.join(self.policy_eval_dir, f"depth/depth_{t}.png"), depth_mm)
                    # save_pointcloud(rgb_bgr, depth_raw, intrinsics, pose, save_path, None)

                    # Save the video from images every #steps_numbers steps
                    steps_numbers = 100
                    if (t+1) % steps_numbers==0:
                        create_video_from_images(
                            img_dir=os.path.join(self.policy_eval_dir, "rgb"), 
                            output_path=os.path.join(self.policy_eval_dir, "trajectory_rgb_video.mp4"),
                            fps=10
                        )
           
                rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                # self.store_filtered_pointcloud(rgb, depth_raw, intrinsics, camera_pose, keep_ratio=1.00, step=t)

                c2w = utils.get_cam_transform(agent_state=self.habitat_ds.sim.sim.get_agent_state()) @ habitat_transform
                c2w_t = torch.from_numpy(c2w).float().cuda()
                w2c_t = torch.linalg.inv(c2w_t)
                
                if self.policy_name == "gaussians_based":
                    ate = slam.track_rgbd(img, depth, w2c_t, action_id)

                    if ate is not None:
                        self.log({"ate": ate}, t)

                if cm.should_exit():
                    cm.requeue()
                
                # Collect 3D agent pose
                agent_pose, _ = utils.get_sim_location(agent_state=self.habitat_ds.sim.sim.get_agent_state())
                self.abs_agent_poses.append(agent_pose)

                # Update habitat vis tool and save the current state
                if self.save_map:
                    if (t+1) % 5 == 0 or t==0:
                        if self.object_scene:
                            self.habvis.save_vis_seen(self.habitat_ds.sim.sim, t, dynamic_scene=self.object_scene, sim_obj=self.sim_obj)
                        else:
                            self.habvis.save_vis_seen(self.habitat_ds.sim.sim, t)

                    self.habvis.update_fow_sim(self.habitat_ds.sim.sim)
                    if self.object_scene:
                        self.habvis.update_obj_sim(self.habitat_ds.sim.sim, self.sim_obj)
                
                # save habvis
                if (slam.cur_frame_idx) % self.slam_config["checkpoint_interval"] == 0 and slam.cur_frame_idx > 0:
                    save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    self.habvis.save(save_path)

                # Analyze which kind of policy we use
                threshold = 0
                object_detected = np.sum(object_mask_bw)
                if object_detected > threshold:
                    
                    self.object_tracking = True # True
                    if not self.init_object_slam_done:   
                        self.init_object_slam = True
                        self.init_object_slam_done = True
                        print(">> Start object tracking and reconstruction")
                        # self.action_queue = queue.Queue(maxsize=100)   # TODO: initialize action queue and put the object in the center of the image
                    else:
                        self.init_object_slam = False
                    
                else:
                    self.object_tracking = False

                if not self.object_tracking:  
                    self.evaluate_3d_object_reconstruction(step=t) 
                    if self.policy_name == "gaussians_based":
                        if (slam.cur_frame_idx) % self.slam_config["checkpoint_interval"] == 0 and slam.cur_frame_idx > 0:
                            save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                            self.policy.save(save_path)
                        
                            # bev_render_pkg = self.policy.render_bev(slam)
                            # bev_render = bev_render_pkg['render'].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format if necessary
                            # bev_render = (bev_render.clip(0., 1.) * 255).astype(np.uint8).copy()
                            # cv2.imwrite(os.path.join(self.policy_eval_dir, f"bev_{slam.cur_frame_idx}.png"), bev_render)
                            # if self.save_data:
                            #     plt.imsave(os.path.join(self.policy_eval_dir, f"bev_{slam.cur_frame_idx}.png"), bev_render)

                        current_agent_pose = slam.get_latest_frame()
                        # print("Current agent pose: ", current_agent_pose)
                        current_agent_pos = current_agent_pose[:3, 3]
                        
                        # update occlusion map
                        self.policy.update_occ_map(depth, c2w_t, t, self.slam_config["downsample_pcd"])

                        # logger.info(f"frame: {slam.cur_frame_idx} Mapping time: {time.time() - mapping_start:.5f}")
                        best_goal = None
                        best_map_path = None
                        best_path = None
                        best_global_path = None

                        while self.action_queue.empty():
                            # pause backend during evaluation
                            slam.pause()

                            if expansion > 10:
                                # replan 10 times, wrong, exit
                                raise NoFrontierError()
                            
                            try:
                                print(">> Planning the best path")
                                best_path, best_map_path, best_goal, best_world_path, \
                                    best_global_path, global_points, EIGs = self.plan_best_path(slam, current_agent_pose, expansion, t, goal_pose)

                                if best_path is None:
                                    logger.warn(f"time_step {t}, no valid path found, re-plan")
                                    continue
                            except PruneException as e:
                                logger.info(" Too many invisible points, replan ... ")
                                continue

                            if best_path is None:
                                print("No best path! Turning")
                                expansion += 1
                                if not self.action_queue.full():
                                    self.action_queue.put(2)
                            else:
                                expansion = 1
                                # Fill into action queue
                                print(best_path)
                                for action_id in best_path:
                                    if not self.action_queue.full():
                                        self.action_queue.put(action_id)
                                    else:
                                        break
                        
                            slam.resume()

                            # visualize map
                            # self.policy.visualize_map(c2w, best_goal, best_map_path, best_global_path)

                            goal_pose = best_goal
                        
                        action_id = self.action_queue.get()
                        # time.sleep(1.)

                    elif self.policy_name == "UPEN":
                        action_id, finish = self.policy.predict_action(t, self.abs_agent_poses, depth)    
                        if finish:
                            t += 1
                            break

                    elif self.policy_name == "TrajReader":
                        pos = self.traj_poses[t, :3]
                        quat = self.traj_poses[t, 3:]

                        set_agent_state(self.habitat_ds.sim.sim, np.concatenate([pos, quat]))

                        observations = None
                        observations = self.habitat_ds.sim.sim.get_sensor_observations()
                        observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}

                        # estimate distance covered by agent
                        current_pos = self.habitat_ds.sim.sim.get_agent_state().position
                        agent_episode_distance += utils.euclidean_distance(current_pos, previous_pos)
                        previous_pos = current_pos
                        t+=1
                        continue
                    
                    elif self.policy_name == "random_walk":
                        if (slam.cur_frame_idx) % self.slam_config["checkpoint_interval"] == 0 and slam.cur_frame_idx > 0 :
                            save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                            self.policy.save(save_path)

                        current_agent_pose = slam.get_latest_frame()
                        current_agent_pos = current_agent_pose[:3, 3]
                        # print("Current agent pose: ", current_agent_pose)
                        # mapping_start = time.time()
                        agent_state = self.habitat_ds.sim._sim.get_agent_state()
                        agent_rotation = agent_state.rotation  # quaternion: x, y, z, w
                        agent_translation = agent_state.position

                        quat = [agent_rotation.w, agent_rotation.x, agent_rotation.y, agent_rotation.z]  # w, x, y, z
                        rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
                        # current_agent_pose = np.eye(4)
                        # current_agent_pose[:3, :3] = rot_matrix
                        # current_agent_pose[:3, 3] = agent_translation

                        # current_agent_pose = slam.get_latest_frame()
                        current_agent_pos = current_agent_pose[:3, 3]


                        # update occlusion map
                        self.policy.update_occ_map(depth, c2w_t, t, self.slam_config["downsample_pcd"])
                        # logger.info(f"Frame: {slam.cur_frame_idx} Mapping time: {time.time() - mapping_start:.5f}")
                        
                        # self.policy.visualize_map(c2w)
                        while self.action_queue.empty():
                            # pause backend during evaluation
                            slam.pause()
                            # Randomly select an action in [1, 2, 3]
                            action_id = np.random.choice([1, 2, 3])
                            self.action_queue.put(action_id)
                            
                            slam.resume()

                            # visualize map
                            # self.policy.visualize_map(c2w, goal_pose, map_path)
                            
                        action_id = self.action_queue.get()

                    elif self.policy_name == "frontier":
                        if (slam.cur_frame_idx) % self.slam_config["checkpoint_interval"] == 0 and slam.cur_frame_idx > 0 :
                            save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                            self.policy.save(save_path)

                        current_agent_pose = camera_pose.copy()
                        
                        # update occlusion map
                        self.policy.update_occ_map(depth, c2w_t, t, self.slam_config["downsample_pcd"])
                        
                        # self.policy.visualize_map(c2w)
                        while self.action_queue.empty():
                            # pause backend during evaluation
                            slam.pause()
                            
                            best_path = None
                            while best_path is None:
                                current_agent_pos = current_agent_pose[:3, 3]

                                global_points, _, _ = \
                                    self.policy.global_planning_frontier(expansion, visualize=True, 
                                                                agent_pose=current_agent_pos)
                                
                                # print("Global points: ", global_points)
                                if global_points is None:
                                    raise NoFrontierError("No frontier found")

                                global_points = global_points.cpu().numpy()
                                
                                # plan actions for each global goal
                                _, path_actions, paths_arr = self.action_planning(global_points, current_agent_pose, None, t)
                                if len(path_actions) == 0:
                                    raise NoFrontierError("No path actions found")

                                best_path = path_actions[0]
                                map_path = paths_arr[0]

                                # print("Best path: ", best_path)


                            if best_path is None:
                                print("No best path! Turning")
                                expansion += 1
                                if not self.action_queue.full():
                                    self.action_queue.put(2)
                            else:
                                expansion = 1
                                # Fill into action queue
                                print(best_path)
                                for action_id in best_path:
                                    if not self.action_queue.full():
                                        self.action_queue.put(action_id)
                                    else:
                                        break
                        
                            # resume backend process after planning
                            slam.resume()

                            # visualize map
                            # self.policy.visualize_map(c2w, goal_pose, map_path)
                            
                        action_id = self.action_queue.get()
                
                # if object tracking is enabled, we use the local policy to track the object
                else:
                    if (slam.cur_frame_idx) % self.slam_config["checkpoint_interval"] == 0 and slam.cur_frame_idx > 0:
                        save_path = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                        self.policy.save(save_path)
                    
                    obj_mask_t = torch.from_numpy(object_mask_bw).bool().cuda()
                    # obj_mask_t = torch.from_numpy(object_mask_bw).unsqueeze(-1).bool().cuda()
                    if self.init_object_slam:
                        # Initialize object SLAM 
                        obj_slam = GaussianObjectSLAM(self.slam_config, cur_frame_idx=slam.cur_frame_idx-1)   # TODO: create a new config for object SLAM
                        obj_slam.init(img, depth, c2w_t, obj_mask_t)
                        self.init_object_policy(obj_slam, c2w, intrinsics, object_mask_bw)

                        # self.init_object_slam_done = True

                    if do_mapping_this_frame:
                        ate_obj = obj_slam.track_rgbd(img, depth, w2c_t, action_id, obj_mask_t, step=t)
                    
                    # Rendering the object
                    # os.makedirs(os.path.join(self.policy_eval_dir, "rendering_object"), exist_ok=True)
                    # image=obj_slam.render_at_pose(c2w_t)
                    # plt.imsave(os.path.join(self.policy_eval_dir, "rendering_object", f"render_mask_{t}.png"), image["render"].permute(1, 2, 0).cpu().numpy().clip(0., 1.))

                    if ate_obj is not None:
                        self.log({"ate_obj": ate_obj}, t)
                            
                    # current_agent_pose = slam.get_latest_frame()
                    current_agent_pose = c2w.copy()
                    # print("Current agent pose: ", current_agent_pose)
                    
                    # update occlusion map
                    self.policy.update_occ_map(depth, c2w_t, t, self.slam_config["downsample_pcd"])

                    # logger.info(f"frame: {slam.cur_frame_idx} Mapping time: {time.time() - mapping_start:.5f}")
                    best_goal = None
                    best_map_path = None
                    best_path = None
                    best_global_path = None

                    while self.action_queue.empty():
                        # pause backend during evaluation
                        slam.pause()
                        # pause object backend during evaluation
                        obj_slam.pause()

                        if expansion > 10:
                            # replan 10 times, wrong, exit
                            raise NoFrontierError()
                        
                        try:
                            print(">> Planning the best path for object reconstruction")
                            # print("Guassian points: ", obj_slam.gaussian_points)
                            best_path, best_map_path, best_goal, best_world_path, \
                                    best_global_path, global_points, EIGs = self.plan_best_object_path(obj_slam, slam, current_agent_pose, expansion, t, goal_pose, criteria=self.slam_config["criterion"])
                            
                            if best_path is None:
                                logger.warn(f"time_step {t}, no valid path found, re-plan")
                                continue
                        except PruneException as e:
                            logger.info(" Too many invisible points, replan ... ")
                            continue

                        if best_path is None:
                            print("No best path! Turning")
                            expansion += 1
                            if not self.action_queue.full():
                                self.action_queue.put(2)
                        else:
                            expansion = 1
                            # Fill into action queue
                            print(best_path)
                            for action_id in best_path:
                                if not self.action_queue.full():
                                    self.action_queue.put(action_id)
                                else:
                                    break
                    
                        slam.resume()

                        # visualize map
                        # self.policy.visualize_map(c2w, best_goal, best_map_path, best_global_path)

                        goal_pose = best_goal
                    
                    action_id = self.action_queue.get()


                # explicitly clear observation otherwise they will be kept in memory the whole time
                observations = None
                
                # Apply next action
                # depth is [0, 1] (should be rescaled to 10)
                prev_pos = self.habitat_ds.sim.sim.get_agent_state().position
                self.habitat_ds.sim.sim.step(action_id)
                current_pos = self.habitat_ds.sim.sim.get_agent_state().position
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
                    while not self.action_queue.empty():
                        action_id = self.action_queue.get()
                    
                    robot_stuck_count += 1
                    if robot_stuck_count > 10:
                        print("Robot stuck for too long, exiting navigation loop")
                        self.options.max_steps = t + 1
                
                # get new observation
                observations = self.habitat_ds.sim.sim.get_sensor_observations()
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
            print("No frontier found, exiting navigation loop")
            pass
        except LocalizationError as e:
            logger.error("Robot inside obstacle")
            pass

        # print("Agent episode distance: ", agent_episode_distance)
        slam.color_refinement()
        self.eval_navigation(slam, t)

        ###### Evaluation of the 3D reconstruction of the scene ######
        self.evaluate_3d_reconstruction()

        if self.slam_config.use_wandb:
            wandb.finish()
        
        # Close current scene
        self.habitat_ds.sim.sim.close()
        # slam.frontend.backend_queue.put(["stop"])
        slam.stop()


    def evaluate_3d_object_reconstruction(self, step=0):
        dist_th = 0.01
        pcl_dir = os.path.join(self.policy_eval_dir, "pointcloud")
        gt_obj_3d_reconstruction = self.gt_obj_3d_rotated

        est_obj_pcl_file = get_latest_pcl_file(pcl_dir, obj=True)
        print("Comparison with GT object model: ", est_obj_pcl_file)
        if est_obj_pcl_file is not None:
            est_obj_3d_reconstruction = load_ply_pointcloud(est_obj_pcl_file)
            est_3d_rotated = apply_transform_to_pointcloud(est_obj_3d_reconstruction, habitat_transform)

            acc, comp, ratio = accuracy_comp_ratio_from_pcl(gt_obj_3d_reconstruction, est_3d_rotated, dist_th=dist_th)
            iacc, icomp, iratio = accuracy_comp_ratio_from_pcl(est_3d_rotated, gt_obj_3d_reconstruction, dist_th=dist_th)

            fpr = (1-iratio)
        else:
            acc = comp = ratio = fpr = 0.0
            iratio = 1.0
        print(f"[eval] step={step} | ACC={acc*100:.2f} cm | COMP={comp*100:.2f} cm | "
          f"Completeness={ratio*100:.2f}% | FPR={fpr*100:.2f}%")

        metrics_dir = os.path.join(self.policy_eval_dir, "metrics")
        yaml_path   = os.path.join(metrics_dir, "object_recon_metrics.yaml")
        from datetime import datetime

        new_row = {
            "step": int(step),
            # "distance_threshold_m": float(dist_th),
            "acc_distance_m": float(acc*100),          # distanza media (accuracy)
            "comp_distance_m": float(comp*100),        # distanza media (completeness)
            "completeness_ratio": float(ratio*100),    # [0..1]
            "fpr": float(fpr*100),                     # [0..1]
            "est_pcl_path": str(est_obj_pcl_file),
        }

        data = yaml_safe_load(yaml_path)
        if data is None:
            data = {
                "experiment": {
                    "policy_name": getattr(self, "policy_name", "unknown"),
                    "scene_id":    getattr(self, "scene_id", "unknown"),
                },
                "settings": {
                    "distance_threshold_m": float(dist_th),
                },
                "steps": [],     # lista di entry per step
                "summary": {},   # riempita in seguito (AUC/finali)
            }

        data["steps"].append(new_row)
        data["steps"] = sorted(data["steps"], key=lambda r: r["step"])
        # ---- AUC (trapezi) aggiornata ad ogni chiamata ----
        xs = np.array([r["step"] for r in data["steps"]], dtype=float)
        ys = np.array([r["completeness_ratio"] for r in data["steps"]], dtype=float)  # percentuale (0..100)

        # opzionale: padding fino a max_steps con plateau dell’ultimo valore
        max_steps = int(data["settings"].get("max_steps", 0))
        if max_steps and len(xs) > 0 and xs[-1] < max_steps:
            xs = np.append(xs, float(max_steps))
            ys = np.append(ys, ys[-2] if len(ys) >= 2 and xs[-2] == max_steps else ys[-2] if len(ys) >= 2 and xs[-2] > xs[-3] else ys[-2] if len(ys) >= 2 else ys[-1])
            # più semplicemente (plateau con ultimo valore):
            ys[-1] = ys[-2] if len(ys) >= 2 else ys[-1]

        # AUC grezza (%·step)
        auc_raw = float(np.trapz(ys, xs)) if len(xs) >= 2 else 0.0
        # AUC normalizzata in [0,1]
        denom = (max_steps * 100.0) if max_steps else ((xs[-1] - xs[0]) * 100.0 if len(xs) >= 2 else 1.0)
        auc_norm = float(auc_raw / denom) if denom > 0 else 0.0

        # salva nel summary (snapshot corrente) e, se vuoi, anche cumulativa per step corrente
        data["summary"]["auc_completeness_percent_vs_steps_raw"] = auc_raw
        data["summary"]["auc_completeness_normalized_0_1"] = auc_norm
        data["summary"]["last_step"] = int(data["steps"][-1]["step"])
        data["summary"]["last_completeness_percent"] = float(data["steps"][-1]["completeness_ratio"])

        # (facoltativo) annota l’AUC cumulativa fino allo step corrente dentro la riga
        new_row["auc_until_now_normalized"] = auc_norm
        yaml_safe_dump(data, yaml_path)


    def evaluate_3d_reconstruction(self):
        gt_3d_reconstruction = load_glb_pointcloud(os.path.join(self.options.root_path, self.options.dataset, self.options.scenes_list[0], self.options.scenes_list[0] + ".glb"))
        # est_3d_reconstruction = load_ply_pointcloud(os.path.join(self.policy_eval_dir, "pointcloud", "global_pcl_{}.ply".format(self.options.max_steps)))
        pcl_dir = os.path.join(self.policy_eval_dir, "pointcloud")
        latest_pcl_file = get_latest_pcl_file(pcl_dir)
        print("Latest scene PCL file: ", latest_pcl_file)
        if latest_pcl_file:
            est_3d_reconstruction = load_ply_pointcloud(latest_pcl_file)
        else:
            raise FileNotFoundError("No valid global_pcl_*.ply file found.")
        
        # save_pointcloud_as_ply(est_3d_reconstruction, os.path.join(self.policy_eval_dir, "original_est_scene_ply.ply"))
        est_3d_rotated = apply_transform_to_pointcloud(est_3d_reconstruction, habitat_transform)
        # save_pointcloud_as_ply(est_3d_rotated, os.path.join(self.policy_eval_dir, "rotated_est_scene_ply.ply"))

        # save_pointcloud_as_ply(gt_3d_reconstruction, os.path.join(self.policy_eval_dir, "original_gt_scene_glb.ply"))
        gt_3d_rotated = apply_transform_to_pointcloud(gt_3d_reconstruction, rotation_90_x)
        # save_pointcloud_as_ply(gt_3d_rotated, os.path.join(self.policy_eval_dir, "rotated_gt_scene_pcl.ply"))

        # coverage = calculate_coverage_percentage(gt_3d_rotated, est_3d_rotated)
        # print(f"#### Scene : {self.options.scenes_list[0]} ####")
        # print(f"Coverage Percentage: {coverage * 100:.2f}%")
        # gt_3d_reconstruction = load_ply_pointcloud("../FisherRF-active-mapping/experiments/GaussianSLAM/GdvgFV5R1Z5-results/MP3D/rotated_gt_scene_pcl.ply")
        # est_3d_reconstruction = load_ply_pointcloud("../FisherRF-active-mapping/experiments/GaussianSLAM/GdvgFV5R1Z5-results/MP3D/rotated_est_scene_ply.ply")
        acc, comp, ratio = accuracy_comp_ratio_from_pcl(gt_3d_rotated, est_3d_rotated, dist_th=0.05)
        iacc, icomp, iratio = accuracy_comp_ratio_from_pcl(est_3d_rotated, gt_3d_rotated, dist_th=0.05)

        print(f"ACC (dist): {acc*100:.2f} cm, Completeness (dist): {comp*100:.2f} cm, Completeness (ratio): {ratio*100:.2f} %, FPR: {(1-iratio)*100:.2f} %")
        with open(os.path.join(self.policy_eval_dir, f"{self.policy_name}_results.txt"), "a") as f:
            f.write(f"Scene Evaluation:\n"
                    f"ACC (dist): {acc*100:.2f} cm, "
                    f"Completeness (dist): {comp*100:.2f} cm, "
                    f"Completeness (ratio): {ratio*100:.2f} %, "
                    f"FPR: {(1-iratio)*100:.2f} %\n")
            
        ##### Evaluation of the 3D reconstruction of the object #####
        if self.dynamic_scene_rec:
            gt_obj_3d_reconstruction = load_glb_pointcloud(os.path.join(self.options.root_path, self.dynamic_object_path, "futuristic_robot_with_single_eye.glb"))
            pcl_dir = os.path.join(self.policy_eval_dir, "pointcloud")
            latest_pcl_file = get_latest_pcl_file(pcl_dir, obj=True)
            print("Latest object PCL file: ", latest_pcl_file)
            if latest_pcl_file:
                est_obj_3d_reconstruction = load_ply_pointcloud(latest_pcl_file)
            else:
                raise FileNotFoundError("No valid global_pcl_*.ply file found.")
            # est_obj_3d_reconstruction = load_ply_pointcloud(os.path.join(self.policy_eval_dir, "pointcloud", "global_pcl_{}.ply".format(self.options.max_steps)))
            
            # save_pointcloud_as_ply(est_obj_3d_reconstruction, "original_ply.ply")
            est_obj_3d_rotated = apply_transform_to_pointcloud(est_obj_3d_reconstruction, habitat_transform)
            # save_pointcloud_as_ply(est_obj_3d_rotated, os.path.join(self.policy_eval_dir, "rotated_est_obj.ply"))

            # save_pointcloud_as_ply(gt_obj_3d_reconstruction, "original_glb.ply")
            gt_obj_3d_rotated = apply_transform_to_pointcloud(gt_obj_3d_reconstruction, rotation_m90_x, scale=OBJ_SCALE_FACTOR)
            # save_pointcloud_as_ply(gt_obj_3d_rotated, os.path.join(self.policy_eval_dir, "rotated_gt_obj.ply"))

            acc, comp, ratio = accuracy_comp_ratio_from_pcl(gt_obj_3d_rotated, est_obj_3d_rotated, dist_th=0.01)
            iacc, icomp, iratio = accuracy_comp_ratio_from_pcl(est_obj_3d_rotated, gt_obj_3d_rotated, dist_th=0.01)

            print(f"ACC (dist): {acc*100:.2f} cm, Completeness (dist): {comp*100:.2f} cm, Completeness (ratio): {ratio*100:.2f} %, FPR: {(1-iratio)*100:.2f} %")
            with open(os.path.join(self.policy_eval_dir, f"{self.policy_name}_results.txt"), "a") as f:
                f.write(f"Object Evaluation:\n"
                        f"ACC (dist): {acc*100:.2f} cm, "
                        f"Completeness (dist): {comp*100:.2f} cm, "
                        f"Completeness (ratio): {ratio*100:.2f} %, "
                        f"FPR: {(1-iratio)*100:.2f} %\n")

    def project_points(self, points_3d, intrinsics, img_h, img_w):
        points_cam = points_3d.T  # shape (3,N)
        points_2d = intrinsics.cpu()[:3,:3] @ points_cam
        points_2d = points_2d[:2] / points_2d[2]

        u = torch.round(points_2d[0]).long()
        # v = np.round(points_2d[1]).astype(int)
        v = torch.round(points_2d[1]).long()

        valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[v[valid], u[valid]] = 255
        return mask

    def uniform_rand_poses(self):
        agent_pose, agent_height = utils.get_sim_location(agent_state=self.habitat_ds.sim.sim.get_agent_state())
        scene_bounds_lower, scene_bounds_upper = self.habitat_ds.sim.sim.pathfinder.get_bounds()

        # Generate Random poses
        test_size = int(2e3)
        rng = np.random.default_rng(42)
        candidate_pos = np.zeros((test_size, 3))
        candidate_pos[:, 0] = rng.uniform(scene_bounds_lower[0], scene_bounds_upper[0], (test_size, ))
        candidate_pos[:, 2] = rng.uniform(scene_bounds_lower[2], scene_bounds_upper[2], (test_size, ))
        candidate_pos[:, 1] = agent_height
        valid_index = list(map(self.habitat_ds.sim.sim.pathfinder.is_navigable, candidate_pos))
        valid_index = np.array(valid_index)
        
        valid_pos = candidate_pos[valid_index]
        
        random_angle = rng.uniform(0., 2 * np.pi, (len(valid_pos), ))
        random_quat = np.zeros((len(valid_pos), 4)) # (in w, x, y, z)
        random_quat[:, 0] = np.cos(random_angle / 2)
        random_quat[:, 2] = np.sin(random_angle / 2)

        return valid_pos, random_quat

    
    @torch.no_grad()
    def eval_navigation(self, slam, log_step: int=0):
        """ The function that really run the evaluation and visualization
        """
        ## Episode ended ##
        slam.pause()
        # print("LOG STEP: ", log_step)
        # PSNR Evaluation
        metrics = {"psnr": [], "depth_mae": [], "ssim": [], "lpips": []}
        init_agent_state = self.habitat_ds.sim.sim.get_agent_state()
        agent_pose, agent_height = utils.get_sim_location(agent_state=self.habitat_ds.sim.sim.get_agent_state())
        scene_bounds_lower, scene_bounds_upper = self.habitat_ds.sim.sim.pathfinder.get_bounds()
        valid_pos, random_quat = self.uniform_rand_poses()
        # print("Valid pose: ", len(valid_pos))
        # print("Random quat: ", len(random_quat))
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")

        
        # os.makedirs(os.path.join(self.policy_eval_dir, "render"), exist_ok=True)
        # os.makedirs(os.path.join(self.policy_eval_dir, "gt"), exist_ok=True)
        # print("Creating directories for eval render and gt")

        gaussians = False
        if gaussians:
            # compute H train
            H_train = slam.compute_H_train()
            H_train_inv = torch.reciprocal(H_train + slam.cfg.H_reg_lambda)

        poses_stats = []

        for test_id, (pos, quat) in tqdm(enumerate(zip(valid_pos, random_quat))):
            set_agent_state(self.habitat_ds.sim.sim, np.concatenate([pos, quat]))

            observations = self.habitat_ds.sim.sim.get_sensor_observations()
            observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}

            # render at position 
            c2w = utils.get_cam_transform(agent_state=self.habitat_ds.sim.sim.get_agent_state()) @ habitat_transform
            c2w_t = torch.from_numpy(c2w).float().cuda()

            with torch.no_grad():
                render_pkg = slam.render_at_pose(c2w_t, white_bg=True)
                w2c = torch.linalg.inv(c2w_t)
                # cur_H, pose_H = self.compute_Hessian( w2c, return_points=True, random_gaussian_params=random_gaussian_params, return_pose=True)
                if gaussians:
                    cur_H, pose_H = slam.compute_Hessian(w2c, return_pose=True, return_points=True)
                    cur_H = cur_H * H_train_inv
                    EIG = torch.log(torch.sum(cur_H * H_train_inv))
                    # set max to 100. (arbitrary choice)
                    if torch.isinf(EIG):
                        EIG = torch.tensor(100.)
                
                color = render_pkg["render"]
                color.clamp_(min=0., max=1.)
                depth = render_pkg["depth"]

            rgb_gt = observations['rgb'][:, :, :3].permute(2, 0, 1) / 255
            depth_gt = observations['depth'].reshape(self.habitat_ds.img_size[0], self.habitat_ds.img_size[1], 1).permute(2, 0, 1)

            color_8bit = color.permute(1, 2, 0).cpu().numpy() * 255

            if gaussians:
                name = "{:06d}.png".format(int(EIG.item() * 1e4))
                plt.figure()
                plt.grid(False)
                plt.imshow(color_8bit.astype(np.uint8))
                plt.title(f"Id: {test_id}, EIG: {EIG.item():.4f}")
                plt.savefig(os.path.join(self.policy_eval_dir, "render", name))
                plt.close() 
            # imageio.imsave(os.path.join(self.policy_eval_dir, "render", f"{test_id}.png"), color_8bit.astype(np.uint8))

            gt_8bit = rgb_gt.permute(1, 2, 0).cpu().numpy() * 255
            # imageio.imsave(os.path.join(self.policy_eval_dir, "gt", f"{test_id}.png"), gt_8bit.astype(np.uint8))

            # compute PSNR & Depth Abs. Error
            psnr = calc_psnr(color, rgb_gt).mean()
            depth_mae = torch.mean(torch.abs( depth - depth_gt ))

            if gaussians:
                poses_stats.append(
                    dict(id=test_id, pose = [p.tolist() for p in c2w], EIG=EIG.item(), psnr=psnr.item())
                )

            ssim_score = ssim((color).unsqueeze(0), (rgb_gt).unsqueeze(0))
            lpips_score = cal_lpips((color).unsqueeze(0), (rgb_gt).unsqueeze(0))

            if torch.isinf(psnr):
                # skip inf psnr, this might due to zero(all black) gt image
                continue

            metrics["psnr"].append(psnr.item())
            metrics["depth_mae"].append(depth_mae.item())
            metrics["ssim"].append(ssim_score.item())
            metrics["lpips"].append(lpips_score.item())

        # export eval poses to json file
        if gaussians:
            with open(os.path.join(self.policy_eval_dir, "eval.json"), "w") as f:
                js.dump(poses_stats, f)

        known_area = torch.tensor(self.habvis.fow_mask).int()
        coord = 0 if known_area.shape[0] < known_area.shape[1] else 1
        meter_per_pixel = min(abs(scene_bounds_upper[c*2] - scene_bounds_lower[c*2]) / known_area.shape[c] for c in [0,1])
        _, semantic_map = draw_map(self.habitat_ds.sim.sim, agent_height, meter_per_pixel, use_sim=True, map_res=known_area.shape[coord])
        gt_know = (semantic_map == 1).astype(np.uint8)

        print("gt_known_area_shape: ", gt_know.shape, "known_area_shape: ", known_area.shape)
        # union = known_area.cpu().numpy() * gt_know

        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(gt_know)
        # axes[1].imshow(known_area.cpu())
        # fig.savefig(os.path.join(self.policy_eval_dir, "area.png"))

        # metrics["coverage(m^2)"] = union.sum() * meter_per_pixel ** 2
        # metrics["coverage(%)"] = union.sum() / gt_know.sum() * 100
        # coverage2d = metrics["coverage(%)"]
        # with open(os.path.join(self.policy_eval_dir, f"{self.policy_name}_results.txt"), "w") as f:
        #     f.write(f"Coverage 2D: {coverage2d:.2f} %")
        
        eval_results = {}
        output = ""
        for k, v in metrics.items():
            m_string = "{}: {:.4f} \n".format(k, np.array(v).mean())
            eval_results[f"test/{k}"] = np.array(v).mean()
            output += m_string
        self.log(eval_results, log_step)

        # with open(os.path.join(self.policy_eval_dir, "results.txt"), "w") as f:  
        #     f.write(output)

        logger.info(output.replace("\n", "\t"))

        meter_per_pixel = 0.05
        top_down_map, _ = draw_map(self.habitat_ds.sim.sim, agent_height, meter_per_pixel)
        cmap = mpl.colormaps["plasma"]
        for pos, psnr in zip(valid_pos, metrics["psnr"]):
            color = list(map(lambda x: int(x * 255), cmap(psnr / 20)[:3]))
            color = np.array(color)

            pixel_x = min( max( int(math.ceil((pos[0] - scene_bounds_lower[0]) / meter_per_pixel)), 0), top_down_map.shape[1] - 1 )
            pixel_z = min( max( int(math.floor((pos[2] - scene_bounds_lower[2]) / meter_per_pixel)), 0), top_down_map.shape[0] - 1 )
            top_down_map[pixel_z, pixel_x] = color

        # for t in range(slam.cur_frame_idx):
        #     # Get the current estimated rotation & translation
        #     curr_w2c = torch.eye(4).cuda()
        #     curr_w2c[:3, :3] = slam.frontend.cameras[t].R
        #     curr_w2c[:3, 3] = slam.frontend.cameras[t].T

        #     curr_c2w = torch.linalg.inv(curr_w2c)
        #     pos = curr_c2w[:3, 3].cpu().numpy()

        #     pixel_x = min( max(int(round((pos[0] - scene_bounds_lower[0]) / meter_per_pixel)), 0), top_down_map.shape[1] - 1 )
        #     pixel_z = min( max(int(round((pos[2] - scene_bounds_lower[2]) / meter_per_pixel)), 0), top_down_map.shape[0] - 1 )
        #     top_down_map[pixel_z, pixel_x] = np.array([0, 255, 0])

        map_filename = os.path.join(self.policy_eval_dir, "top_down_eval_viz.png")
        imageio.imsave(map_filename, top_down_map)

        self.habitat_ds.sim.sim.agents[0].set_state(init_agent_state)
        # Close current scene

    @staticmethod
    def render_sim_at_pose(sim, c2w):
        set_agent_state(sim, c2w)

        observations = sim.get_sensor_observations()
        # sim_obs = self.habitat_ds.sim.sim.get_sensor_observations()
        # observations = self.habitat_ds.sim.sim._sensor_suite.get_observations(sim_obs)
        image_size = observations["rgb"].shape[:2]  

        color = observations["rgb"][:, :, :3].permute(2, 0, 1) / 255
        depth = observations['depth'].reshape(image_size[0], image_size[1], 1)

        return color, depth
    
    def add_pose_noise(self, rel_pose, action_id):
        if action_id == 1:
            x_err, y_err, o_err = self.habitat_ds.sensor_noise_fwd.sample()[0][0]
        elif action_id == 2:
            x_err, y_err, o_err = self.habitat_ds.sensor_noise_left.sample()[0][0]
        elif action_id == 3:
            x_err, y_err, o_err = self.habitat_ds.sensor_noise_right.sample()[0][0]
        else:
            x_err, y_err, o_err = 0., 0., 0.
        rel_pose[0,0] += x_err*self.options.noise_level
        rel_pose[0,1] += y_err*self.options.noise_level
        rel_pose[0,2] += torch.tensor(np.deg2rad(o_err*self.options.noise_level))
        return rel_pose

    def log(self, output, log_step=0):
        for k in output:
            self.summary_writer.add_scalar(k, output[k], log_step)

        if self.slam_config.use_wandb:
            wandb.log(output, self.step_count)

    def plan_best_path(self, slam: GaussianSLAM, 
                       current_agent_pose: np.array, 
                       expansion:int,  t: int, last_goal = None):
        """ Path & Action planning 
        
        Args:
            slam -- Gaussian SLAM system.
            current_agent_pose (4, 4) 
            expansion (int) -- expansion factor for sampling pose
            t (int) --  time step
            last_goal (np.array) -- last goal point

        Return:
            best_path, 
            best_map_path, 
            best_goal,             (4, 4) selected goal point
            best_world_path, 
            best_global_path         
            global_point:           (N, 4, 4) global points
            EIGs:                   (N, ) EIG for each goal point
        """
        current_agent_pos = current_agent_pose[:3, 3]
        gaussian_points = slam.gaussian_points
        
        # global plan -- select global 
        pose_proposal_fn = None if not hasattr(slam, "pose_proposal") else getattr(slam, "pose_proposal")
        # print("Pose proposal function: ", pose_proposal_fn)
        global_points, EIGs, random_gaussian_params = \
            self.policy.global_planning(slam.pose_eval, gaussian_points, pose_proposal_fn, \
                                        expansion=expansion, visualize=True, \
                                        agent_pose=current_agent_pos, last_goal=last_goal, slam=slam)

        # sort global points by EIG 
        EIGs = EIGs.numpy()
        global_points = global_points.cpu().numpy()
        sort_index = np.argsort(EIGs)[::-1]
        global_points = global_points[sort_index]
        EIGs = EIGs[sort_index]
        
        if self.cfg.num_uniform_H_train > 0:
            uiform_positions, uiform_quants = self.uniform_rand_poses()
            H_train = None
            for cur_uni_pos, cur_uni_quat in tqdm(random.sample(list(zip(uiform_positions, uiform_quants)), \
                                                                self.cfg.num_uniform_H_train), 
                                                                desc="Computing uniformH_train"):
                cur_uni_w2c = pos_quant2w2c(cur_uni_pos, cur_uni_quat, self.habitat_ds.sim.sim.get_agent_state())

                cur_H = slam.compute_Hessian(cur_uni_w2c, random_gaussian_params=random_gaussian_params, return_points=True, return_pose=False)
                H_train = H_train + cur_H if H_train is not None else cur_H.detach().clone()
        else:
            H_train = slam.compute_H_train(random_gaussian_params)
            # H_train = rearrange(H_train, "np c -> (np c)")
        
        gs_pts_cnt = slam.gs_pts_cnt(random_gaussian_params)

        best_global_path = None
        best_path_EIG = -1.
        best_path = None
        best_goal = None
        best_map_path = None
        best_world_path = None
        valid_path = 0

        # plan actions for each global goal
        valid_global_pose, path_actions, paths_arr = self.action_planning(global_points, current_agent_pose, slam.gaussian_points, t)
        logger.info(f"Evaluate path actions: {len(path_actions)}")
        total_path_EIGs = []

        for pose_np, path_action, paths, final_EIG in tqdm(zip(valid_global_pose, path_actions, paths_arr, EIGs), desc="Evaluate Paths"):
            # check cluster manager  
            if cm.should_exit():
                cm.requeue()

            if valid_path > 20:
                break
            
            valid_path += 1
           
            # set to cam height
            future_pose = current_agent_pose.copy()
            future_pose[1, 3] = self.policy.cam_height

            H_train_path = H_train.clone()
            total_path_EIG = 0
            map_path = []
            world_path = []
            curr_action = []

            for action in path_action:

                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])
                future_pose_w2c = np.linalg.inv(future_pose)
                cur_H, pose_H = slam.compute_Hessian(future_pose_w2c, random_gaussian_params=random_gaussian_params, 
                                                        return_pose=True, return_points=True)
                H_train_inv_path = torch.reciprocal(H_train_path + self.cfg.H_reg_lambda)
                
                if self.cfg.vol_weighted_H:
                    point_EIG = torch.log(torch.sum(cur_H * H_train_inv_path / gs_pts_cnt))
                else:
                    point_EIG = torch.log(torch.sum(cur_H * H_train_inv_path))
                
                pose_EIG = torch.log(torch.linalg.det(pose_H))

                curr_action.append(action)
            
                total_path_EIG += self.cfg.path_pose_weight * pose_EIG.item()
                # iterative cumulation
                if (len(curr_action) + 1) % self.cfg.acc_H_train_every == 0:
                    total_path_EIG += self.cfg.path_point_weight * point_EIG.item()
                    H_train_path = H_train_path + cur_H

                if action == 1:
                    map_coord = future_pose[[0, 2], 3]
                    world_path.append(map_coord)
                    map_coord = self.policy.convert_to_map(map_coord)
                    map_path.append(map_coord)

            if self.cfg.path_end_weight > 0:
                total_path_EIG = total_path_EIG / len(curr_action) + self.cfg.path_end_weight * final_EIG
            else:
                total_path_EIG = (total_path_EIG + final_EIG) / len(curr_action)
            
            total_path_EIGs.append(total_path_EIG)

            # select the best one
            if total_path_EIG > best_path_EIG:
                best_path_EIG = total_path_EIG
                best_path = curr_action
                best_goal = pose_np
                best_global_path = paths
                best_map_path = map_path
                best_world_path = world_path

        # dump paths_arr and total_path_EIGs using pickle
        # with open(os.path.join(self.policy.eval_dir, f"paths_arr_{t}.pkl"), "wb") as f:
        #     stats = dict(paths_arr=paths_arr, total_path_EIGs=total_path_EIGs,
        #                  map_center = self.policy.map_center.cpu().numpy(), cell_size = self.policy.cell_size,
        #                  grid_dim = self.policy.grid_dim)
        #     pickle.dump(paths_arr, stats)

        return best_path, best_map_path, best_goal, best_world_path, best_global_path, global_points, EIGs

    def plan_best_object_path(self, obj_slam: GaussianObjectSLAM, slam: GaussianSLAM, 
                       current_agent_pose: np.array, 
                       expansion:int,  t: int, last_goal = None, criteria=None):
        """ Path & Action planning 
        
        Args:
            slam -- Gaussian SLAM system.
            current_agent_pose (4, 4) 
            expansion (int) -- expansion factor for sampling pose
            t (int) --  time step
            last_goal (np.array) -- last goal point

        Return:
            best_path, 
            best_map_path, 
            best_goal,             (4, 4) selected goal point
            best_world_path, 
            best_global_path         
            global_point:           (N, 4, 4) global points
            EIGs:                   (N, ) EIG for each goal point
        """
        current_agent_pos = current_agent_pose[:3, 3]
        gaussian_points = obj_slam.gaussian_points
        gaussian_points_scene = slam.gaussian_points
        
        # global plan -- select global 
        pose_proposal_fn = None if not hasattr(obj_slam, "pose_proposal") else getattr(obj_slam, "pose_proposal")
        print("Start global planning for object")
        if criteria.lower() == "fisher":
            global_points, EIGs, random_gaussian_params, candidate_object_pose = \
                self.policy.global_object_planning(obj_slam.pose_eval, gaussian_points, gaussian_points_scene, pose_proposal_fn, \
                                            expansion=expansion, visualize=True, \
                                            agent_pose=current_agent_pos)
        else:
            global_points, EIGs, random_gaussian_params, candidate_object_pose = \
                self.policy.global_object_planning(obj_slam.pose_eval_popgs, gaussian_points, gaussian_points_scene, pose_proposal_fn, \
                                            expansion=expansion, visualize=True, \
                                            agent_pose=current_agent_pos, criterion=criteria)
        

        
        print(f"Found {len(global_points)} global points: ")
        # sort global points by EIG 
        EIGs = EIGs.numpy()
        global_points = global_points.cpu().numpy()
        sort_index = np.argsort(EIGs)[::-1]
        global_points = global_points[sort_index]
        EIGs = EIGs[sort_index]
        print("Sorted global points by EIG: ", EIGs[:10])
        if self.cfg.num_uniform_H_train > 0:
            uiform_positions, uiform_quants = self.uniform_rand_poses()
            H_train = None
            for cur_uni_pos, cur_uni_quat in tqdm(random.sample(list(zip(uiform_positions, uiform_quants)), \
                                                                self.cfg.num_uniform_H_train), 
                                                                desc="Computing uniformH_train"):
                cur_uni_w2c = pos_quant2w2c(cur_uni_pos, cur_uni_quat, self.habitat_ds.sim.sim.get_agent_state())

                cur_H = obj_slam.compute_Hessian(cur_uni_w2c, random_gaussian_params=random_gaussian_params, return_points=True, return_pose=False)
                H_train = H_train + cur_H if H_train is not None else cur_H.detach().clone()
        else:
            if criteria.lower() == "fisher":
                H_train = obj_slam.compute_H_train(random_gaussian_params)
            else:
                H_train = obj_slam.compute_H_train_popgs()
            # H_train_diag = obj_slam.compute_H_train_popgs()
        
        gs_pts_cnt = obj_slam.gs_pts_cnt(random_gaussian_params)

        # plan actions for each global goal
        valid_global_pose, path_actions, paths_arr = self.action_planning_object_adv(global_points, current_agent_pose, slam.gaussian_points, t)
        logger.info(f"Evaluate path actions: {len(path_actions)}")

        if criteria.lower() == "fisher":
            best_path, best_map_path, best_goal, best_world_path, best_global_path = self.path_evaluation(valid_global_pose, path_actions, paths_arr, EIGs, current_agent_pose, H_train, random_gaussian_params, obj_slam)
            # best_path, best_map_path, best_goal, best_world_path, best_global_path = self.path_object_evaluation(valid_global_pose, path_actions, paths_arr, EIGs, current_agent_pose, H_train, random_gaussian_params, candidate_object_pose, obj_slam)
        else:
            best_path, best_map_path, best_goal, best_world_path, best_global_path = self.path_evaluation_popgs(valid_global_pose, path_actions, paths_arr, EIGs, current_agent_pose, H_train, random_gaussian_params, obj_slam, criterion=criteria)

        self.draw_final_map(best_goal, current_agent_pos, best_global_path, t)

        return best_path, best_map_path, best_goal, best_world_path, best_global_path, global_points, EIGs

    def draw_final_map(self, best_goal, current_agent_pos, best_global_path, t=0):
        occ_map = self.policy.occ_map.argmax(0) == 1
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
        if self.policy.frontier.sum() != 0:
            frontier = self.policy.frontier.copy()
            frontier = cv2.dilate(frontier.astype(np.uint8), kernel, iterations=1) 
            vis_map[:,:,0][frontier!=0] = 0
            vis_map[:,:,1][frontier!=0] = 255
            vis_map[:,:,2][frontier!=0] = 0

        # best poses
        best_pose = best_goal  # shape (4,4)

        # posizione in mondo -> pixel mappa
        best_pt = self.policy.convert_to_map([best_pose[0, 3].item(), best_pose[2, 3].item()])

        # cv2.circle(vis_map, (best_pt[0], best_pt[1]), 3, (0, 255, 255), -1)  # fill
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
        end_world_x = best_pose[0, 3].item() + dir_x * (arrow_len * self.policy.cell_size)
        end_world_z = best_pose[2, 3].item() + dir_z * (arrow_len * self.policy.cell_size)
        end = self.policy.convert_to_map([end_world_x, end_world_z])

        cv2.arrowedLine(vis_map, start, (end[0], end[1]), (0, 255, 255), 2, tipLength=0.35) 

        # agent position
        pt = self.policy.convert_to_map([current_agent_pos[0], current_agent_pos[2]])
        cv2.circle(vis_map, (pt[0], pt[1]), 4, (255, 0, 0), 2) 
        vis_map = cv2.circle(vis_map, (pt[0],pt[1]), 2, (255,0,0), -1)

        # Waypoint position
        for waypoint in best_global_path:
            cv2.circle(vis_map, (waypoint[0], waypoint[1]), 2, (255, 255, 0), 2) 
            vis_map = cv2.circle(vis_map, (waypoint[0],waypoint[1]), 1, (255,255,0), -1)

        os.makedirs(os.path.join(self.policy.eval_dir, "final_maps"), exist_ok=True)
        plt.imsave(os.path.join(self.policy.eval_dir, "final_maps", "occmap_with_candidates_{}.png".format(t)), vis_map)
        plt.close()

    def path_evaluation(self, valid_global_pose, path_actions, paths_arr, EIGs, current_agent_pose, H_train, random_gaussian_params, obj_slam: GaussianObjectSLAM):
        
        total_path_EIGs = []

        best_global_path = None
        best_path_EIG = -1.
        best_path = None
        best_goal = None
        best_map_path = None
        best_world_path = None
        valid_path = 0

        gs_pts_cnt = obj_slam.gs_pts_cnt(random_gaussian_params)

        for pose_np, path_action, paths, final_EIG in tqdm(zip(valid_global_pose, path_actions, paths_arr, EIGs), desc="Evaluate Paths"):
            # check cluster manager  
            if cm.should_exit():
                cm.requeue()

            if valid_path > 20:
                break
            
            valid_path += 1
           
            # set to cam height
            future_pose = current_agent_pose.copy()
            future_pose[1, 3] = self.policy.cam_height

            H_train_path = H_train.clone()
            total_path_EIG = 0
            map_path = []
            world_path = []
            curr_action = []

            for action in path_action:

                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])
                future_pose_w2c = np.linalg.inv(future_pose)
                cur_H, pose_H, vis_count = obj_slam.compute_Hessian(future_pose_w2c, random_gaussian_params=random_gaussian_params, 
                                                        return_pose=True, return_points=True)
                H_train_inv_path = torch.reciprocal(H_train_path + self.cfg.H_reg_lambda)
                
                if self.cfg.vol_weighted_H:
                    if vis_count==0:
                        point_EIG = torch.tensor(0.)
                    else:
                        point_EIG = torch.log(torch.sum(cur_H * H_train_inv_path / gs_pts_cnt))
                    point_EIG = torch.tensor(0.) 
                else:
                    # If no visible points, set point_EIG to 0 or set a penalty
                    if vis_count==0:
                        point_EIG = torch.tensor(0.) 
                    else:
                        point_EIG = torch.log(torch.sum(cur_H * H_train_inv_path))
                    point_EIG = torch.tensor(0.) 
                
                pose_EIG = torch.log(torch.linalg.det(pose_H))

                curr_action.append(action)
            
                total_path_EIG += self.cfg.path_pose_weight * pose_EIG.item()
                # iterative cumulation
                if (len(curr_action) + 1) % self.cfg.acc_H_train_every == 0:
                    total_path_EIG += self.cfg.path_point_weight * point_EIG.item()
                    H_train_path = H_train_path + cur_H

                if action == 1:
                    map_coord = future_pose[[0, 2], 3]
                    world_path.append(map_coord)
                    map_coord = self.policy.convert_to_map(map_coord)
                    map_path.append(map_coord)

            if self.cfg.object_path_end_weight > 0:
                # total_path_EIG = total_path_EIG / len(curr_action) + self.cfg.object_path_end_weight * final_EIG
                total_path_EIG = total_path_EIG + self.cfg.object_path_end_weight * final_EIG
            else:
                total_path_EIG = (total_path_EIG + final_EIG) / len(curr_action)
            
            total_path_EIGs.append(total_path_EIG)

            # select the best one
            if total_path_EIG > best_path_EIG:
                best_path_EIG = total_path_EIG
                best_path = curr_action
                best_goal = pose_np
                best_global_path = paths
                best_map_path = map_path
                best_world_path = world_path

        return best_path, best_map_path, best_goal, best_world_path, best_global_path

    def path_object_evaluation(self, valid_global_pose, path_actions, paths_arr, EIGs, current_agent_pose, H_train, random_gaussian_params, candidate_object_pose, obj_slam: GaussianObjectSLAM):
        
        total_path_EIGs = []
        
        best_global_path = None
        best_path_EIG = -1.
        best_path = None
        best_goal = None
        best_map_path = None
        best_world_path = None
        valid_path = 0

        probe_every = 5
        max_queue = 200

        gs_pts_cnt = obj_slam.gs_pts_cnt(random_gaussian_params)

        def yaw_of_pose(T):
            return np.arctan2(T[0, 2], T[2, 2])

        def angle_wrap(a):
            return np.arctan2(np.sin(a), np.cos(a))
        
        for pose_np, path_action, paths, final_EIG in tqdm(zip(valid_global_pose, path_actions, paths_arr, EIGs), desc="Evaluate Paths"):
            # check cluster manager  
            if cm.should_exit():
                cm.requeue()

            if valid_path > 20:
                break
            
            valid_path += 1
           
            # set to cam height
            future_pose = current_agent_pose.copy()
            future_pose[1, 3] = self.policy.cam_height

            # Set the object candidate pose
            # temp_object_pose = np.array([candidate_object_pose[0], future_pose[1, 3], candidate_object_pose[1], 1.0])

            obj_np = candidate_object_pose.detach().cpu().numpy().reshape(-1)  # -> array([x, z])
            temp_object_pose = np.array([obj_np[0], future_pose[1, 3], obj_np[1], 1.0], dtype=np.float32)
            
            H_train_path = H_train.clone()
            total_path_EIG = 0
            map_path = []
            world_path = []
            curr_action = []

            steps_since_probe = 0

            def eval_info_at_pose(future_pose_4x4):
                future_w2c = np.linalg.inv(future_pose_4x4)
                cur_H, pose_H, vis_count = obj_slam.compute_Hessian(
                    future_w2c,
                    random_gaussian_params=random_gaussian_params,
                    return_pose=True,
                    return_points=True
                )
                H_inv = torch.reciprocal(H_train_path + self.cfg.H_reg_lambda)
                if self.cfg.vol_weighted_H:
                    if vis_count == 0:
                        point_EIG = torch.tensor(0.0)
                    else:
                        point_EIG = torch.log(torch.sum(cur_H * H_inv / gs_pts_cnt))
                else:
                    point_EIG = torch.tensor(0.0) if vis_count == 0 else torch.log(torch.sum(cur_H * H_inv))

                pose_EIG = torch.log(torch.linalg.det(pose_H))
                return point_EIG, pose_EIG, cur_H, vis_count
            
            for action in path_action:

                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])
                curr_action.append(action)
                
                # Evaluate info gain at the new pose
                point_EIG, pose_EIG, cur_H, vis_count = eval_info_at_pose(future_pose)
                if vis_count == 0:
                    steps_since_probe += 1

                # Every N steps test small rotations to observe the object, if a rotation improves the info gain, take it
                if steps_since_probe >= probe_every and len(curr_action) < max_queue:
                    steps_since_probe = 0
                    
                    # yaw_now  = yaw_of_pose(future_pose)
                    while abs(ang_wp) > abs(np.radians(self.slam_config["turn_angle"])) and len(curr_action) < max_queue:
                        rel_wp = np.linalg.inv(future_pose) @ temp_object_pose
                        ang_wp = np.arctan2(rel_wp[0], rel_wp[2])
                        if ang_wp >  np.radians(self.slam_config["turn_angle"]):
                            act = 3  # turn right
                        elif ang_wp < -np.radians(self.slam_config["turn_angle"]):
                            act = 2  # turn left
                        else:
                            break
                        
                        future_pose = compute_next_campos(future_pose, act, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])

                    point_EIG, pose_EIG, cur_H, _ = eval_info_at_pose(future_pose)

                total_path_EIG += self.cfg.path_pose_weight * pose_EIG.item()
                # iterative cumulation
                if (len(curr_action) + 1) % self.cfg.acc_H_train_every == 0:
                    total_path_EIG += self.cfg.path_point_weight * point_EIG.item()
                    H_train_path = H_train_path + cur_H

                if action == 1:
                    map_coord = future_pose[[0, 2], 3]
                    world_path.append(map_coord)
                    map_coord = self.policy.convert_to_map(map_coord)
                    map_path.append(map_coord)

            if self.cfg.object_path_end_weight > 0:
                # total_path_EIG = total_path_EIG / len(curr_action) + self.cfg.object_path_end_weight * final_EIG
                total_path_EIG = total_path_EIG + self.cfg.object_path_end_weight * final_EIG
            else:
                total_path_EIG = (total_path_EIG + final_EIG) / len(curr_action)
            
            total_path_EIGs.append(total_path_EIG)

            # select the best one
            if total_path_EIG > best_path_EIG:
                best_path_EIG = total_path_EIG
                best_path = curr_action
                best_goal = pose_np
                best_global_path = paths
                best_map_path = map_path
                best_world_path = world_path

        return best_path, best_map_path, best_goal, best_world_path, best_global_path


    def path_evaluation_popgs(self, valid_global_pose, path_actions, paths_arr, EIGs, current_agent_pose, H_train_diag, random_gaussian_params, obj_slam: GaussianObjectSLAM, criterion="topt", lam=1e-6):

        total_path_EIGs = []

        best_global_path = None
        best_path_EIG = -float("inf")
        best_path = None
        best_goal = None
        best_map_path = None
        best_world_path = None
        valid_path = 0

        for pose_np, path_action, paths, final_EIG in tqdm(zip(valid_global_pose, path_actions, paths_arr, EIGs), desc="Evaluate Paths"):
            # check cluster manager  
            if cm.should_exit():
                cm.requeue()

            if valid_path > 20:
                break
            
            valid_path += 1
           
            # set to cam height
            future_pose = current_agent_pose.copy()
            future_pose[1, 3] = self.policy.cam_height

            H_train_path = H_train_diag.clone()
            total_path_EIG = 0
            map_path = []
            world_path = []
            curr_action = []

            for action in path_action:

                future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])
                future_pose_w2c = np.linalg.inv(future_pose)
                # cur_H, pose_H, vis_count = obj_slam.compute_Hessian(future_pose_w2c, random_gaussian_params=random_gaussian_params, 
                #                                         return_pose=True, return_points=True)
                cur_diag, vis_count = obj_slam.estimate_diag_JtJ_simple(future_pose_w2c)
                # H_train_inv_path = torch.reciprocal(H_train_path + self.cfg.H_reg_lambda)
                # Hpi = H_train_path + cur_diag + lam
                Hm  = H_train_path + lam
                Hpi = Hm + cur_diag
                if self.cfg.vol_weighted_H:
                    if vis_count==0:
                        point_EIG = torch.tensor(0.)
                    else:
                        if criterion.lower() == "topt":
                            point_EIG = - torch.sum(1.0 / torch.clamp(Hpi, min=1e-12))
                        elif criterion.lower() == "dopt":
                            point_EIG = torch.sum(torch.log(torch.clamp(Hpi, min=1e-12))) - torch.sum(torch.log(torch.clamp(Hm,  min=1e-12)))
                else:
                    # If no visible points, set point_EIG to 0 or set a penalty
                    if vis_count==0:
                        point_EIG = torch.tensor(0.) 
                    else:
                        if criterion.lower() == "topt":
                            point_EIG = - torch.sum(1.0 / torch.clamp(Hpi, min=1e-12))
                        elif criterion.lower() == "dopt":
                            point_EIG = torch.sum(torch.log(torch.clamp(Hpi, min=1e-12))) - torch.sum(torch.log(torch.clamp(Hm,  min=1e-12)))
                
                # pose_EIG = torch.log(torch.linalg.det(pose_H))

                curr_action.append(action)
            
                # total_path_EIG += self.cfg.path_pose_weight * pose_EIG.item()
                # iterative cumulation
                if (len(curr_action) + 1) % self.cfg.acc_H_train_every == 0:
                    total_path_EIG += float(self.cfg.path_point_weight) * float(point_EIG.item())
                    H_train_path = H_train_path + cur_diag

                if action == 1:
                    map_coord = future_pose[[0, 2], 3]
                    world_path.append(map_coord)
                    map_coord = self.policy.convert_to_map(map_coord)
                    map_path.append(map_coord)

            final_EIG_f = float(final_EIG.item() if hasattr(final_EIG, "item") else final_EIG)

            if self.cfg.path_end_weight > 0:
                total_path_EIG = total_path_EIG / len(curr_action) + float(self.cfg.object_path_end_weight) * final_EIG_f
            else:
                total_path_EIG = (total_path_EIG + final_EIG_f) / len(curr_action)

            total_path_EIGs.append(total_path_EIG)

            # select the best one
            if total_path_EIG > best_path_EIG:
                best_path_EIG = total_path_EIG
                best_path = curr_action
                best_goal = pose_np
                best_global_path = paths
                best_map_path = map_path
                best_world_path = world_path

        return best_path, best_map_path, best_goal, best_world_path, best_global_path


    def action_planning(self, global_points, current_agent_pose, gaussian_points, t):
        """
        Plan sequences of actions for each goal poses

        Args:
            global_points (np.array): (N, 4, 4) goal poses
            current_agent_pose (np.array): (4, 4) current agent pose
            gaussian_points: Gaussian Points from MonoGS
            t: time step

        Return:
            valid_global_points (List[np.array]): valid goal poses
            path_actions: List[List[int]]: action for each goal
            paths_arr: List[] List for each planned path
        """
        valid_global_points = []
        path_actions = []
        paths_arr = []

        # set start position in A* Planner
        current_agent_pos = current_agent_pose[:3, 3]
        start = self.policy.convert_to_map(current_agent_pos[[0, 2]])[[1, 0]] # from x-z to z-x
        self.policy.setup_start(start, gaussian_points, t)

        for pose_np in tqdm(global_points, desc="Action Planning"):
            # print("Planning for pose: ", pose_np)
            if cm.should_exit():
                cm.requeue()

            pos_np = pose_np[:3, 3].copy()
            pos_np[1] = current_agent_pos[1] # set the same heights

            finish = self.policy.convert_to_map(pos_np[[0, 2]])[[1, 0]] # convert to z-x
            # print("Finish: ", finish)

            paths = self.policy.planning(finish) # A* Planning in [x, z] TODO: try also other planning methods
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
            # print("Stage goal: ", stage_goal, "Stage goal world: ", stage_goal_w)
            # push actions to queue
            while len(path_action) < self.slam_config["policy"]["planning_queue_size"]:
                # compute action
                rel_pos = np.linalg.inv(future_pose) @ stage_goal_w
                xz_rel_pos = rel_pos[[0, 2]]
                # print("Distance to stage goal: ", np.linalg.norm(xz_rel_pos), "Stage goal idx: ", stage_goal_idx)
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
                        stage_goal_w = self.policy.convert_to_world(stage_goal+0.5)
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

    def action_planning_object_adv(self, global_points, current_agent_pose, gaussian_points, t):
        """
        Plan sequences of actions for each goal pose.

        Args:
            global_points (np.array): (N, 4, 4) goal poses
            current_agent_pose (np.array): (4, 4) current agent pose
            gaussian_points: Gaussian Points from MonoGS
            t: time step

        Return:
            valid_global_points (List[np.array]): valid goal poses
            path_actions: List[List[int]]: actions for each goal
            paths_arr: List[np.ndarray]: grid waypoints for each goal
        """
        # ---------- Tunable tolerances ----------
        step = self.slam_config["forward_step_size"]
        turn_deg = self.slam_config["turn_angle"]
        turn = np.radians(turn_deg)

        # if we're this close to the final position, switch to orientation-only
        POS_TOL_FINAL = 2.5 * step
        # yaw tolerance to consider “aligned enough”
        YAW_TOL_FINAL = turn
        # if a waypoint is this close to the final goal, skip it
        SKIP_WP_IF_NEAR_GOAL = 2.0 * step
        # if the next waypoint does not reduce distance-to-goal by this margin, skip it
        SKIP_WP_MARGIN = 0.25 * step
        # safety cap if we run long
        SAFETY_CAP = 200

        def yaw_of_pose(T):
            return np.arctan2(T[0, 2], T[2, 2])

        def angle_wrap(a):
            return np.arctan2(np.sin(a), np.cos(a))

        def world_from_grid_cell_center(cell_zy):
            w = self.policy.convert_to_world(cell_zy + 0.5)  # (x,z)
            return np.array([w[0], future_pose[1, 3], w[1], 1.0])

        valid_global_points, path_actions, paths_arr = [], [], []

        # set start position in A* Planner
        start = self.policy.convert_to_map(current_agent_pose[[0, 2], 3])[[1, 0]]  # z,x
        self.policy.setup_start(start, gaussian_points, t)

        for pose_np in tqdm(global_points, desc="Action Planning"):
            if cm.should_exit():
                cm.requeue()

            # build A* path on the grid to goal position (ignore goal height here)
            goal_pos = pose_np[:3, 3].copy()
            goal_pos[1] = current_agent_pose[1, 3]
            finish = self.policy.convert_to_map(goal_pos[[0, 2]])[[1, 0]]  # z,x

            path_grid = self.policy.planning(finish)  # (M,2) in z,x
            if len(path_grid) == 0:
                continue

            # If only one cell returned, add finish to ensure at least one move is possible
            if len(path_grid) == 1:
                # path_grid = np.concatenate([path_grid, finish[None, :]], axis=0)
                if len(path_grid) == 1:
                    if not np.array_equal(path_grid[0], finish):
                        path_grid = np.concatenate([path_grid, finish[None, :]], axis=0)
                    else:
                        path_grid = np.concatenate([path_grid, path_grid[0][None, :]], axis=0)

            # ---------- prune waypoints that are "too near" the final goal ----------
            pruned = []
            goal_world_4 = np.array([pose_np[0, 3], current_agent_pose[1, 3], pose_np[2, 3], 1.0])
            for p in path_grid:
                wpt_world_4 = np.array([*self.policy.convert_to_world(p + 0.5), current_agent_pose[1, 3], 1.0])  # x,z,y,1
                d_goal = np.linalg.norm((wpt_world_4[[0, 2]] - goal_world_4[[0, 2]]))
                if d_goal > SKIP_WP_IF_NEAR_GOAL:
                    pruned.append(p)
            if len(pruned) == 0:
                pruned = [path_grid[0], path_grid[-1]]  # keep at least start/end
            path_grid = np.array(pruned, dtype=np.int32)
            if path_grid.shape[0] < 2:
                path_grid = np.vstack([path_grid, finish[None, :]])

            # ---------- simulate actions ----------
            future_pose = current_agent_pose.copy()
            future_pose[1, 3] = self.policy.cam_height
            stage_idx = 1
            stage_goal = path_grid[stage_idx]
            stage_goal_w4 = world_from_grid_cell_center(stage_goal)

            acts = []
            used_steps = 0

            while used_steps < SAFETY_CAP: # and len(acts) < self.slam_config["policy"]["planning_queue_size"]:
                # distance to final goal in camera frame
                final_goal_w4 = np.array([pose_np[0, 3], future_pose[1, 3], pose_np[2, 3], 1.0])
                rel_final = np.linalg.inv(future_pose) @ final_goal_w4
                d_final = np.linalg.norm(rel_final[[0, 2]])
                yaw_now = yaw_of_pose(future_pose)
                yaw_goal = yaw_of_pose(pose_np)
                dyaw = angle_wrap(yaw_goal - yaw_now)

                # ---------- early-stop if "good enough" ----------
                if (d_final < POS_TOL_FINAL) and (abs(dyaw) <= YAW_TOL_FINAL):
                    break

                # ---------- orientation-only mode near final ----------
                if d_final < POS_TOL_FINAL:
                    act = 2 if dyaw > 0 else 3
                    # if within one turn step, done
                    if abs(dyaw) <= YAW_TOL_FINAL:
                        break
                    future_pose = compute_next_campos(future_pose, act, step, turn_deg)
                    acts.append(act)
                    used_steps += 1
                    continue

                # ---------- otherwise: move along waypoints ----------
                # if close enough to current stage waypoint, advance
                rel_wp = np.linalg.inv(future_pose) @ stage_goal_w4
                d_wp = np.linalg.norm(rel_wp[[0, 2]])
                if d_wp < step:
                    # try to skip to the last waypoint if the next is almost redundant
                    if stage_idx + 1 < len(path_grid):
                        next_wp = path_grid[stage_idx + 1]
                        next_wp_w4 = world_from_grid_cell_center(next_wp)
                        rel_next = np.linalg.inv(future_pose) @ next_wp_w4
                        rel_goal = np.linalg.inv(future_pose) @ final_goal_w4
                        # if next waypoint is not improving the approach enough → skip
                        if (np.linalg.norm(rel_goal[[0, 2]]) - np.linalg.norm(rel_next[[0, 2]])) < SKIP_WP_MARGIN:
                            # skip directly to goal (position)
                            stage_goal_w4 = final_goal_w4
                            stage_idx = len(path_grid) - 1
                        else:
                            stage_idx += 1
                            stage_goal = path_grid[stage_idx]
                            stage_goal_w4 = world_from_grid_cell_center(stage_goal)
                    else:
                        # last waypoint reached → switch to final goal
                        stage_goal_w4 = final_goal_w4

                    continue  # re-evaluate with updated stage_goal_w4

                # Compute steering to current stage waypoint
                rel_wp = np.linalg.inv(future_pose) @ stage_goal_w4
                ang_wp = np.arctan2(rel_wp[0], rel_wp[2])

                # yaw priority when significantly misaligned
                if ang_wp >  turn:
                    act = 3  # turn right
                elif ang_wp < -turn:
                    act = 2  # turn left
                else:
                    act = 1  # forward

                future_pose = compute_next_campos(future_pose, act, step, turn_deg)
                acts.append(act)
                used_steps += 1

            if acts not in path_actions and len(acts) > 0:
                path_actions.append(acts)
                valid_global_points.append(pose_np)
                paths_arr.append(path_grid)

        return valid_global_points, path_actions, paths_arr

    def action_planning_object(self, global_points, current_agent_pose, gaussian_points, t):
        """
        Plan sequences of actions for each goal poses

        Args:
            global_points (np.array): (N, 4, 4) goal poses
            current_agent_pose (np.array): (4, 4) current agent pose
            gaussian_points: Gaussian Points from MonoGS
            t: time step

        Return:
            valid_global_points (List[np.array]): valid goal poses
            path_actions: List[List[int]]: action for each goal
            paths_arr: List[] List for each planned path
        """
        valid_global_points = []
        path_actions = []
        paths_arr = []

        # set start position in A* Planner
        current_agent_pos = current_agent_pose[:3, 3]
        start = self.policy.convert_to_map(current_agent_pos[[0, 2]])[[1, 0]] # from x-z to z-x
        self.policy.setup_start(start, gaussian_points, t)
        
        for pose_np in tqdm(global_points, desc="Action Planning"):
            # print("Planning for pose: ", pose_np)
            if cm.should_exit():
                cm.requeue()

            pos_np = pose_np[:3, 3].copy()
            pos_np[1] = current_agent_pos[1] # set the same heights

            finish = self.policy.convert_to_map(pos_np[[0, 2]])[[1, 0]] # convert to z-x
            
            # Paths include the waypoints to the goal
            paths = self.policy.planning(finish) # A* Planning in [x, z] TODO: try also other planning methods
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
            
            stage_goal_w = self.policy.convert_to_world(stage_goal+0.5) # grid cell center
            stage_goal_w = np.array([stage_goal_w[0], future_pose[1, 3], stage_goal_w[1], 1])
            path_action = []
            generate_opposite_turn = False
            # print("Stage goal: ", stage_goal, "Stage goal world: ", stage_goal_w)
            # push actions to queue
            while len(path_action) < self.slam_config["policy"]["planning_queue_size"]:
                # compute action
                
                rel_pos = np.linalg.inv(future_pose) @ stage_goal_w
                xz_rel_pos = rel_pos[[0, 2]]
                # print("Distance to stage goal: ", np.linalg.norm(xz_rel_pos), "Stage goal idx: ", stage_goal_idx)
                # Check if close the the final target pose
                pose_np_w = pose_np[[0, 2], 3]  # x,z finali
                final_goal_w = np.array([pose_np_w[0], future_pose[1, 3], pose_np_w[1], 1.0])

                rel_final = np.linalg.inv(future_pose) @ final_goal_w
                xz_rel_final = rel_final[[0, 2]]
                dist_to_final = np.linalg.norm(xz_rel_final)

                # Se sei "abbastanza vicino" alla POSIZIONE finale → passa a modalità ORIENTAZIONE-ONLY
                orientation_only = dist_to_final < self.slam_config["forward_step_size"]*2.5

                if orientation_only:
                    # errore di yaw alla posa finale
                    yaw_now  = np.arctan2(future_pose[0, 2], future_pose[2, 2])
                    yaw_goal = np.arctan2(pose_np[0, 2],    pose_np[2, 2])
                    dyaw = np.arctan2(np.sin(yaw_goal - yaw_now), np.cos(yaw_goal - yaw_now))
                    dyaw_deg = np.degrees(dyaw)

                    if abs(dyaw_deg) <= np.radians(self.slam_config["turn_angle"]):
                        # allineato a sufficienza → esci
                        break

                    # Altrimenti accumula SOLO azioni di rotazione verso il segno di dyaw
                    action = 2 if dyaw > 0 else 3    # + → sinistra, - → destra
                    future_pose = compute_next_campos(
                        future_pose, action,
                        self.slam_config["forward_step_size"], self.slam_config["turn_angle"]
                    )
                    path_action.append(action)
                    continue  # salta il resto e ricalcola dyaw nel prossimo ciclo

                else:
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
                            stage_goal_w = self.policy.convert_to_world(stage_goal+0.5)
                            stage_goal_w = np.array([stage_goal_w[0], future_pose[1, 3], stage_goal_w[1], 1])
                            rel_pos = np.linalg.inv(future_pose) @ stage_goal_w
                            xz_rel_pos = rel_pos[[0, 2]]
                        
                    angle = np.arctan2(xz_rel_pos[0], xz_rel_pos[1])
                    # print("Angle to stage goal: ", np.rad2deg(angle))
                    if angle > np.radians(self.slam_config["turn_angle"]):
                        action = 3
                    elif angle < - np.radians(self.slam_config["turn_angle"]):
                        action = 2
                    else:
                        action = 1
                        
                    future_pose = compute_next_campos(future_pose, action, self.slam_config["forward_step_size"], self.slam_config["turn_angle"])
                    path_action.append(action)
            
            # ---------- POST-LOOP: se budget esaurito, completa i waypoint mancanti; poi final approach ----------
            if len(path_action) >= self.slam_config["policy"]["planning_queue_size"]:
                step = self.slam_config["forward_step_size"]
                turn = np.radians(self.slam_config["turn_angle"])
                SAFETY_CAP = 100

                extra = 0
                while extra < SAFETY_CAP:
                    # 1) Se ci sono ancora waypoint da raggiungere, continua a seguirli
                    if stage_goal_idx < len(paths):
                        # print("Stage goal idx missing: ", stage_goal_idx, " of ", len(paths))
                        # usa SEMPRE il centro cella del waypoint
                        stage_goal = paths[stage_goal_idx]
                        sg_w = self.policy.convert_to_world(stage_goal + 0.5)
                        stage_goal_w = np.array([sg_w[0], future_pose[1, 3], sg_w[1], 1.0])

                        rel = np.linalg.inv(future_pose) @ stage_goal_w
                        xz_rel = rel[[0, 2]]

                        # Check if close the the final target pose
                        pose_np_w = pose_np[[0, 2], 3]  # x,z finali
                        final_goal_w = np.array([pose_np_w[0], future_pose[1, 3], pose_np_w[1], 1.0])

                        rel_final = np.linalg.inv(future_pose) @ final_goal_w
                        xz_rel_final = rel_final[[0, 2]]
                        dist_to_final = np.linalg.norm(xz_rel_final)

                        # Se sei "abbastanza vicino" alla POSIZIONE finale → passa a modalità ORIENTAZIONE-ONLY
                        orientation_only = dist_to_final < self.slam_config["forward_step_size"]*2.5

                        if orientation_only:
                            # errore di yaw alla posa finale
                            yaw_now  = np.arctan2(future_pose[0, 2], future_pose[2, 2])
                            yaw_goal = np.arctan2(pose_np[0, 2],    pose_np[2, 2])
                            dyaw = np.arctan2(np.sin(yaw_goal - yaw_now), np.cos(yaw_goal - yaw_now))
                            dyaw_deg = np.degrees(dyaw)

                            if abs(dyaw) <= np.radians(self.slam_config["turn_angle"]):
                                # allineato a sufficienza → esci
                                break

                            # Altrimenti accumula SOLO azioni di rotazione verso il segno di dyaw
                            act = 2 if dyaw > 0 else 3    # + → sinistra, - → destra
                            # future_pose = compute_next_campos(
                            #     future_pose, action,
                            #     self.slam_config["forward_step_size"], self.slam_config["turn_angle"]
                            # )
                            # path_action.append(action)
                            # continue  # salta il resto e ricalcola dyaw nel prossimo ciclo
                        else:
                            d_to_wp = np.linalg.norm(xz_rel)
                            # print("Distance to waypoint: ", d_to_wp)
                            if d_to_wp < step:
                                stage_goal_idx += 1
                                continue  # passa al prossimo waypoint
                            else:
                                ang = np.arctan2(xz_rel[0], xz_rel[1])
                                # print("Angle to waypoint: ", np.rad2deg(ang))
                                act = 3 if ang >  turn else (2 if ang < -turn else 1)
                    # 2) Altrimenti (tutti i waypoint fatti), final approach alla posa target
                    else:
                        # distanza planare e differenza yaw verso la posa target
                        dist = np.linalg.norm(future_pose[[0, 2], 3] - pose_np[[0, 2], 3])
                        pose_np_w = pose_np[[0, 2], 3] # x, z
                        final_goal_w = np.array([pose_np_w[0], future_pose[1, 3], pose_np_w[1], 1.0])
                        
                        yaw_now  = np.arctan2(future_pose[0, 2], future_pose[2, 2])
                        yaw_goal = np.arctan2(pose_np[0, 2],    pose_np[2, 2])
                        dyaw = np.arctan2(np.sin(yaw_goal - yaw_now), np.cos(yaw_goal - yaw_now))

                        rel = np.linalg.inv(future_pose) @ final_goal_w
                        xz_rel = rel[[0, 2]]
                        ang = np.arctan2(xz_rel[0], xz_rel[1])
                        d_to_wp = np.linalg.norm(xz_rel)
                        # print("GO TO THE TARGET POSE")
                        # print("Distance to final goal init: ", dist)
                        # print("Distance to final goal: ", d_to_wp)
                        # print("Distance to target yaw init : ", np.rad2deg(dyaw))
                        # print("Distance to target yaw : ", np.rad2deg(ang))
                        # criterio di uscita: vicino e allineato
                        if (dist < 2.0 * step) and (abs(dyaw) <= turn):
                            break

                        target_w = np.array([pose_np[0, 3], future_pose[1, 3], pose_np[2, 3], 1.0])
                        rel = np.linalg.inv(future_pose) @ target_w
                        xz_rel = rel[[0, 2]]
                        ang = np.arctan2(xz_rel[0], xz_rel[1])
                        # print("Angle to FINAL goal: ", np.rad2deg(ang))
                        # priorità: allineati verso il target, poi avanza
                        act = 2 if dyaw >  turn else (3 if dyaw < -turn else 1)

                    # esegui l’azione scelta
                    future_pose = compute_next_campos(future_pose, act, step, np.degrees(turn))
                    path_action.append(act)
                    extra += 1

            if path_action not in path_actions:
                path_actions.append(path_action)
                valid_global_points.append(pose_np)
                paths_arr.append(paths)
        
        return valid_global_points, path_actions, paths_arr

    def load_3d_gaussian(self, slam:GaussianSLAM, weight_file):
        time_idx = int(weight_file.split('/')[-1].split('.')[0][6:])
       
        logger.info(f"Loading Checkpoint for Frame {weight_file}")
        params_np = dict(np.load(weight_file, allow_pickle=True))
        params = {k: torch.tensor(params_np[k]).cuda().float().requires_grad_(True) for k in params_np.keys() if k not in ["Uncertainty", "occ_map"]}
        
        slam.variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        slam.variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        slam.variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        slam.variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        slam.params = params
        
        # Load the keyframe time idx list
        slam.keyframe_time_indices = np.load(os.path.join(slam.eval_dir, f"keyframe_time_indices{time_idx}.npy"))
        slam.keyframe_time_indices = slam.keyframe_time_indices.tolist()

        slam.frame_idx = time_idx 
        
        # Update the ground truth poses list
        for t in range(slam.frame_idx + 1):
            # Get the current estimated rotation & translation
            curr_cam_rot = F.normalize(slam.params['cam_unnorm_rots'][..., t].detach())
            curr_cam_tran = slam.params['cam_trans'][..., t].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran

            curr_c2w = torch.linalg.inv(curr_w2c)
            set_agent_state(self.habitat_ds.sim.sim, curr_c2w)

            observations = None
            observations = self.habitat_ds.sim.sim.get_sensor_observations()
            observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}
            # sim_obs = self.habitat_ds.sim.get_sensor_observations()
            # observations = self.habitat_ds.sim._sensor_suite.get_observations(sim_obs)

            color = observations["rgb"][:, :, :3].permute(2, 0, 1) / 255
            depth = observations['depth'].reshape(self.habitat_ds.img_size[0], self.habitat_ds.img_size[1], 1)

            if t in slam.keyframe_time_indices:
                # depth = utils.unnormalize_depth(depth.clone(), min=self.min_depth, max=self.max_depth)
                curr_keyframe = {'id': t, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                
                # Add to keyframe list
                slam.keyframe_list.append(curr_keyframe)

            # Update UPEN
            if self.slam_config["policy"]["name"] == "UPEN":
                agent_pose, y_height = utils.get_sim_location(agent_state=self.habitat_ds.sim.get_agent_state())
                self.abs_agent_poses.append(agent_pose)

                self.policy.predict_action(t, self.abs_agent_poses, depth) 
            # Update iSDF model here
            elif self.slam_config["policy"]["name"] == "iSDF":
                pass
    

    def init_local_policy(self, slam,
                        init_c2w,
                        intrinsics,
                        episode = None, known_env_mode=False) -> queue.Queue:
        """
        Init the local policy

        slam: the Gaussian SLAM system
        init_c2w: the initial camera pose
        intrinsics: the camera intrinsics
        episode: episode information, set to None
        """
        self.action_queue = queue.Queue(maxsize=100)
        
        if self.policy_name in ["gaussians_based", "frontier", "random_walk"]:
            # if self.known_env_mode: 
            #     self.policy.init_known_env_from_known_env(init_c2w, self.gt_3d_oriented_w)
            # else:
            #     self.policy.init(init_c2w, intrinsics)
            self.policy.init(init_c2w, intrinsics)
            print("Init Astar Policy: ", self.policy_name)
            if slam.cur_frame_idx > 0:
                folder = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
                ckpt_path = os.path.join(folder, "astar.pth")
                print("Loading Astar Policy from: ", ckpt_path)
                self.policy.load(ckpt_path)
                t = slam.cur_frame_idx
            else:
                # turn around for initialization # Usually 72 steps
                init_scan_steps = 9 if not self.options.debug else 2
                # for k in range(2):
                for k in range(init_scan_steps):
                    self.action_queue.put(2)
            # self.action_queue = queue.Queue(maxsize=self.slam_config["policy"]["planning_queue_size"])
        
        elif self.policy_name == "UPEN":
            self.policy.init(self.habitat_ds, episode)

            if slam.cur_frame_idx > 0:
                slam.pause()
                
                # load trajectory
                traj_path = os.path.join(folder, "traj.npz")
                traj = np.load(traj_path)["est_traj"]

                for curr_w2c in tqdm(traj, desc="Loading Traj ..."):
                    curr_c2w = np.linalg.inv(curr_w2c)
                    set_agent_state(self.habitat_ds.sim.sim, curr_c2w)

                    observations = None
                    observations = self.habitat_ds.sim.sim.get_sensor_observations()
                    observations = {"rgb": torch.from_numpy(observations["rgb"]).cuda(), "depth": torch.from_numpy(observations["depth"]).cuda()}
                    depth = observations['depth'].reshape(1, self.habitat_ds.img_size[0], self.habitat_ds.img_size[1])

                    agent_pose, y_height = utils.get_sim_location(agent_state=self.habitat_ds.sim.sim.get_agent_state())
                    self.abs_agent_poses.append(agent_pose)
                    self.policy.predict_action(t, self.abs_agent_poses, depth) 
                    t += 1

                slam.resume()

        # reset habvis
        self.habvis.reset()
        habvis_size = 768 if not hasattr(self, "policy") else self.policy.grid_dim[0]
        if self.object_scene:
            self.habvis.set_map(self.habitat_ds.sim.sim, habvis_size, dynamic_scene=self.object_scene, sim_obj=self.sim_obj)
        else:
            self.habvis.set_map(self.habitat_ds.sim.sim, habvis_size)
        # load from checkpoint
        if slam.cur_frame_idx > 0:
            folder = os.path.join(slam.save_dir, "point_cloud/iteration_step_{}".format(slam.cur_frame_idx))
            self.habvis.load(folder)
        
        return self.action_queue
    
    def init_object_policy(self,
                       obj_slam,           # GaussianObjectSLAM
                       init_c2w,           # camera pose attuale (c2w)
                       intrinsics,         # torch 3x3
                       object_mask_bw=None,# numpy HxW (opzionale per centratura)
                       episode=None) -> queue.Queue:
        """
        Init della local policy per la ricostruzione dell'oggetto.
        Ritorna una nuova self.action_queue dedicata all'oggetto.
        """
        
        # self.action_queue = queue.Queue(maxsize=100)
        if self.action_queue is None:
            self.action_queue = queue.Queue(maxsize=100)
            self.policy.init(init_c2w, intrinsics)

        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break
        # Initialize the object SLAM
        # 
        print("Init Object Policy")

        if obj_slam.cur_frame_idx > 0:
            folder = os.path.join(obj_slam.save_dir, "object_point_cloud/iteration_step_{}".format(obj_slam.cur_frame_idx))
            ckpt_path = os.path.join(folder, "astar_obj.pth")
            if os.path.exists(ckpt_path):
                print("Loading Object Astar Policy from:", ckpt_path)
                self.policy.load(ckpt_path)
        else:
            # Check mask
            if object_mask_bw is not None:
                err = object_center_error(object_mask_bw)   # [-1,1]
                actions = []

                if err is not None:
                    if err > 0.30:
                        actions.extend([3, 3])
                    elif err > 0.15:
                        actions.append(3)
                    elif err > 0.5:
                        actions.extend([3, 3, 3])
                    elif err < -0.30:
                        actions.extend([2, 2])
                    elif err < -0.15:
                        actions.append(2)
                    elif err < -0.5:
                        actions.extend([2, 2, 2])

                # Fill the action queue: first centering actions, then orbit
                for a in actions:
                    if not self.action_queue.full():
                        self.action_queue.put(a)
        
        if self.action_queue is None:
            self.habvis.reset()
            habvis_size = 768 if not hasattr(self, "policy") else self.policy.grid_dim[0]
            if self.object_scene:
                self.habvis.set_map(self.habitat_ds.sim.sim, habvis_size, dynamic_scene=self.object_scene, sim_obj=self.sim_obj)
            else:
                self.habvis.set_map(self.habitat_ds.sim.sim, habvis_size)

        # 6) Parametri SLAM più aggressivi per l’oggetto (se non già impostati)
        # if hasattr(obj_slam, "config"):
        #     obj_slam.config['keyframe_every'] = 1
        #     obj_slam.config['map_every'] = 1
        #     obj_slam.config['mapping_window_size'] = max(3, obj_slam.config.get('mapping_window_size', 5))

        return self.action_queue


