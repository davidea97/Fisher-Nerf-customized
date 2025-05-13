from habitat.utils.visualizations import maps, fog_of_war

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

import matplotlib.pyplot as plt

import numpy as np
import cv2
import scipy.ndimage

import os
from typing import Any, Dict, Tuple
from datasets.util.utils import depth_to_3D, get_cam_transform

from habitat.utils.visualizations import utils

import imageio

OBJECT_SPRITE = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "robot_icon.png",
    )
)

VALUE_COLORS = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8), cv2.COLORMAP_INFERNO
)
TARGET_FRONTIER_COLOR = (255,0,0) #red
FRONTIER_COLOR = (255,0,255) #pink
SELECTED_PT_COLOR = (0,255,0) #green
TARGET_FRONTIER_COLOR = (255,0,0) #red
FRONTIER_COLOR = (255,0,255) #pink
SELECTED_PT_COLOR = (0,255,0) #green
DEFAULT_PT_COLOR = (0,0,255) #blue
SELECTED_PATH_COLOR = 200 #red-ish. Note this is a habitat color. In range 10 to 255 it will follow cv2 jet cmap

FRONTIER_SIZE =2
PT_SIZE = 4
PT_SELECT_SIZE = 4

SELECTED_PATH_SIZE = 3
TRAJECTORY_SIZE = 3


class HabitatVisualizer:
    def __init__(self, save_dir,scene_id):
        #manually set map_res to same as astar for now 
        #because it's not set until policy is initialized...
        self.save_dir = save_dir + '/pathviz/'

        os.makedirs(self.save_dir,exist_ok=True)

        self.map = None
        self.fow_mask = None



        self._min_height = 0.1 
        self._max_height = 1.2

        rotated_scenes = ["Eastville", "Elmira", "Swormville", "Ribera"]

        self.rotate_map = scene_id in rotated_scenes

    def reset(self):
        self.map = None
        self.fow_mask = None


    def set_map(self, sim, map_res=768, dynamic_scene=False, sim_obj=None):
        self.map_res = map_res
        self.map = maps.get_topdown_map_from_sim(
            sim,
            map_resolution = self.map_res,
            draw_border = True,
            meters_per_pixel = None,
            agent_id = 0,
        )

        if self.rotate_map:
            self.map = maps.get_topdown_map_from_sim(
                sim,
                map_resolution = self.map.shape[1],
                draw_border = True,
                meters_per_pixel = None,
                agent_id = 0,
            )

        self.fow_mask = np.zeros_like(self.map)

        agent_state = sim.get_agent_state()
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self.map.shape[0], self.map.shape[1]),
            sim=sim,
        )

        self.prev_points = [(a_x, a_y)]
        
        if dynamic_scene:
            obj_translation = sim_obj.get_translation()  # This is a numpy array (x, y, z)
            a_x_obj, a_y_obj = maps.to_grid(
                obj_translation[2],
                obj_translation[0],
                (self.map.shape[0], self.map.shape[1]),
                sim=sim,
            )
            self.prev_obj_points = [(a_x_obj, a_y_obj)]

    def magnum_to_numpy_quat(self, mq):
        # mq: _magnum.Quaternion
        import quaternion
        return quaternion.quaternion(
            mq.scalar,          # this is w
            mq.vector.x,        # this is x
            mq.vector.y,        # this is y
            mq.vector.z         # this is z
        )
    
    def get_pos_rot_obj(self, sim, sim_obj):
        obj_translation = sim_obj.get_translation()  # This is a numpy array (x, y, z)
        obj_rotation = sim_obj.obj.rotation  # This is a quaternion (x, y, z, w)+
        
        obj_quat_rotation = self.magnum_to_numpy_quat(obj_rotation)

        a_x, a_y = maps.to_grid(
            obj_translation[2],
            obj_translation[0],
            (self.map.shape[0], self.map.shape[1]),
            sim=sim,
        )
        heading_vector = quaternion_rotate_vector(
            obj_quat_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]
        angle = np.array(phi)

        return np.array([a_x, a_y]), angle

    def get_pos_rot(self, sim):
        agent_state = sim.get_agent_state()
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self.map.shape[0], self.map.shape[1]),
            sim=sim,
        )

        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]
        angle = np.array(phi)

        return np.array([a_x, a_y]), angle

    def update_fow(self, pos, angle, viz_dist=100):
        new_mask = fog_of_war.reveal_fog_of_war(
            top_down_map=self.map,
            current_fog_of_war_mask=self.fow_mask,
            current_point=pos,
            current_angle=angle,
            fov = 90,
            max_line_len = viz_dist,
        )

        self.fow_mask = np.clip(self.fow_mask+new_mask, 0,1)

        self.prev_points += [(pos[0],pos[1])]

    def update_obj_traj(self, pos):
        self.prev_obj_points += [(pos[0],pos[1])]

    def update_fow_sim(self, sim):
        pos, angle = self.get_pos_rot(sim)

        viz_dist = 10.0 / maps.calculate_meters_per_pixel(
                    self.map_res, sim=sim
                )

        self.update_fow(pos, angle, viz_dist)

    def update_obj_sim(self, sim, sim_obj):
        pos, _ = self.get_pos_rot_obj(sim, sim_obj)

        self.update_obj_traj(pos)

    def save(self,save_path):
        np.savez(os.path.join(save_path, "habvis"), 
        mask=self.fow_mask, 
        prev_points=self.prev_points,
        map=self.map)

    def load(self,path):
        checkpoint = np.load(os.path.join(path,"habvis.npz"))
        self.fow_mask = checkpoint["mask"]
        self.prev_points = checkpoint["prev_points"]
        self.map = checkpoint["map"]

    def save(self,save_path):
        np.savez(os.path.join(save_path, "habvis"), 
        mask=self.fow_mask, 
        prev_points=self.prev_points,
        map=self.map)

    def load(self,path):
        checkpoint = np.load(os.path.join(path,"habvis.npz"))
        self.fow_mask = checkpoint["mask"]
        self.prev_points = checkpoint["prev_points"]
        self.map = checkpoint["map"]

        
    def colorize_draw_agent_and_fit_to_height(
        self, topdown_map_info: Dict[str, Any], output_height: int
    ):
        top_down_map = topdown_map_info["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, topdown_map_info["fog_of_war_mask"]
        )

        for agent_idx in range(len(topdown_map_info["map_coord"])):
            map_agent_pos = topdown_map_info["map_coord"][agent_idx]
            map_agent_angle = topdown_map_info["angle"][agent_idx]
            agent_type = topdown_map_info["type"][agent_idx]
            if agent_type == "agent":
                top_down_map = maps.draw_agent(
                    image=top_down_map,
                    agent_center_coord=map_agent_pos,
                    agent_rotation=map_agent_angle,
                    agent_radius_px=min(top_down_map.shape[0:2]) // 32,
                )
            else:
                top_down_map = self.draw_object(
                    image=top_down_map,
                    obj_center_coord=map_agent_pos,
                    obj_rotation=map_agent_angle,
                    obj_radius_px=min(top_down_map.shape[0:2]) // 32,
                )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        old_h, old_w, _ = top_down_map.shape
        top_down_height = output_height
        top_down_width = int(float(top_down_height) / old_h * old_w)

        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )

        return top_down_map

    def draw_object(
        self,
        image: np.ndarray,
        obj_center_coord: Tuple[int, int],
        obj_rotation: float,
        obj_radius_px: int = 5,
    ) -> np.ndarray:
        """
        Draws a rotated object sprite onto the map image.
        """
        rotated_obj = scipy.ndimage.interpolation.rotate(
            OBJECT_SPRITE, obj_rotation * 180 / np.pi
        )

        initial_obj_size = OBJECT_SPRITE.shape[0]
        new_size = rotated_obj.shape[0]
        obj_size_px = max(
            1, int(obj_radius_px * 2 * new_size / initial_obj_size)
        )

        resized_obj = cv2.resize(
            rotated_obj,
            (obj_size_px, obj_size_px),
            interpolation=cv2.INTER_LINEAR,
        )

        utils.paste_overlapping_image(image, resized_obj, obj_center_coord)

        return image
        
    def save_vis_seen(self, sim, t, candidate_pts=None, pt_scores=None, selected_pt=None, selected_path=None, frontier=None, target_frontier=None, dynamic_scene=False, sim_obj=None):
        """
        candidate_pts: should be a list of poses in world frame
        pt_scores: should be the scores associated to candidate_pts, 
            where a higher score is bad (not required even if candidate_pts is given)
        selected_pt: is the pose selected as the best (not required even if candidate_pts is given)
        selected_path: should be a list of (x,y) points in the world frame
        frontier: should be list of (x,y) points in the world frame
        """
        vis_map = self.map.copy()
        # vis_map_obj = self.map.copy()


        pos,rot = self.get_pos_rot(sim)

        maps.draw_path(
                top_down_map= vis_map,
                path_points=self.prev_points,
                color = 10,
                thickness = TRAJECTORY_SIZE,
            )
        
        if dynamic_scene:
            
            maps.draw_path(
                top_down_map=vis_map,
                path_points=self.prev_obj_points,
                color = 200,
                thickness = TRAJECTORY_SIZE,
            )

            pos_obj, rot_obj = self.get_pos_rot_obj(sim, sim_obj)
            info_dict_obj = {
                "map": vis_map,
                "fog_of_war_mask": self.fow_mask,
                "type": ["agent", "object"],
                "map_coord": [pos, pos_obj],
                "angle": [rot, rot_obj],
            }
            
            vis_map = self.colorize_draw_agent_and_fit_to_height(info_dict_obj, output_height=self.map_res)
        else:

            info_dict = {"map": vis_map,
                    "fog_of_war_mask": self.fow_mask,
                    "agent_map_coord": [pos],
                    "agent_angle": [rot]}
            vis_map = maps.colorize_draw_agent_and_fit_to_height(info_dict, output_height=self.map_res)
        
        if self.rotate_map:
            vis_map = cv2.rotate(vis_map, cv2.ROTATE_90_CLOCKWISE)
            # vis_map_obj = cv2.rotate(vis_map_obj, cv2.ROTATE_90_CLOCKWISE)

        if not frontier is None:
            for pt in frontier:
                a_x, a_y = maps.to_grid(
                    pt[1],
                    pt[0],
                    (vis_map.shape[0], vis_map.shape[1]),
                    sim=sim,
                )
                vis_map = cv2.circle(vis_map, (a_y,a_x), FRONTIER_SIZE, FRONTIER_COLOR, -1)
                
        if not target_frontier is None:
            for pt in target_frontier:
                a_x, a_y = maps.to_grid(
                    pt[1],
                    pt[0],
                    (vis_map.shape[0], vis_map.shape[1]),
                    sim=sim,
                )
                vis_map = cv2.circle(vis_map, (a_y,a_x), FRONTIER_SIZE, TARGET_FRONTIER_COLOR, -1)

        if not candidate_pts is None:
            if not pt_scores is None:
                pt_scores_arr = np.array(pt_scores)
                sort_index = np.argsort(pt_scores_arr)[::-1]
                pt_scores_scaled = np.empty_like(sort_index)
                pt_scores_scaled[sort_index] = np.arange(len(pt_scores))
                if len(pt_scores) != 256:
                    pt_scores_scaled = pt_scores_scaled/255
                # pt_scores_max = np.max(pt_scores_arr)
                # pt_scores_min = np.min(pt_scores_arr)
                # pt_scores_scaled = (pt_scores_arr-pt_scores_min)/(pt_scores_max-pt_scores_min)
                # pt_scores_scaled *=255
            c = DEFAULT_PT_COLOR
            for i in range(len(candidate_pts)):
                if not pt_scores is None:
                    ca = VALUE_COLORS[int(pt_scores_scaled[i])][0]
                    c = (int(ca[2]),int(ca[1]),int(ca[0])) 

                pt_xyz = candidate_pts[i][:3,3]

                a_x, a_y = maps.to_grid(
                    pt_xyz[2],
                    pt_xyz[0],
                    (vis_map.shape[0], vis_map.shape[1]),
                    sim=sim,
                )
                    
                vis_map = cv2.circle(vis_map, (a_y,a_x), PT_SIZE, c, -1)

        if not selected_pt is None:
            c = SELECTED_PT_COLOR
            a_x, a_y = maps.to_grid(
                selected_pt[2,3],
                selected_pt[0,3],
                (vis_map.shape[0], vis_map.shape[1]),
                sim=sim,
            )
            vis_map = cv2.circle(vis_map, (a_y,a_x), PT_SELECT_SIZE, c, 2)

        if not selected_path is None:
            path_points = []
            for i in range(len(selected_path)):
                xy = selected_path[i]
                a_x, a_y = maps.to_grid(
                    xy[1],
                    xy[0],
                    (vis_map.shape[0], vis_map.shape[1]),
                    sim=sim,
                )
                path_points += [(a_x,a_y)]

            maps.draw_path(
                top_down_map= vis_map,
                path_points=path_points,
                color = SELECTED_PATH_COLOR,
                thickness = SELECTED_PATH_SIZE,
            )
            
        plt.imsave(self.save_dir + f'gt_pos_path_{t}.png', vis_map)
        # plt.imsave(self.save_dir + f'gt_pos_path_obj_{t}.png', vis_map_obj)
        plt.close()


