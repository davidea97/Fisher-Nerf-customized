
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