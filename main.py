
# import matplotlib
# matplotlib.use('TKAgg')
# From
# https://stackoverflow.com/questions/55811545/importerror-cannot-load-backend-tkagg-which-requires-the-tk-interactive-fra
import torch.multiprocessing as mp
# mp.set_sharing_strategy('file_descriptor')
mp.set_sharing_strategy('file_system')

import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
print(f"Adding project root {project_root} to sys.path")
sys.path.insert(0, project_root)

# Get the root of habitat (adjust this if needed)
habitat_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(habitat_root)

from train_options import TrainOptions
from tester_gaussians_navigation import NavTester

def nav_testing(options, scene_id, object_scene=True, dynamic_scene=False, dynamic_scene_rec=False, dino_extraction=False, save_data=False, save_map=False, known_env=None):
    tester = NavTester(options, scene_id, object_scene, dynamic_scene, dynamic_scene_rec, dino_extraction, save_data, save_map, known_env)
    tester.test_navigation()

if __name__ == '__main__':
    __spec__ = None
    options = TrainOptions().parse_args()
    options.dataset_type = options.dataset
    options.config_val_file = os.path.join("configs", f"my_pointnav_{options.dataset.lower()}_val.yaml")
    scene_ids = options.scenes_list
    dynamic_scene = False
    object_scene = True
    dynamic_scene_rec = True
    dino_extraction = False
    save_data = True
    save_map = True
    known_env_flag = True

    if known_env_flag == True:
        known_envs = []
        for scene_id in scene_ids:
            known_env = os.path.join(project_root, options.root_path, options.dataset_type, scene_id, scene_id + ".glb")
            known_envs.append(known_env)
    else:
        known_envs = [None] * len(scene_ids)
    
    # Create iterables for map function
    n = len(scene_ids)
    options_list = [options] * n
    args = [*zip(options_list, scene_ids)]
    nav_testing(*args[0], object_scene, dynamic_scene, dynamic_scene_rec, dino_extraction, save_data, save_map, known_envs[0])
    
    print("Now the pool is closed and no longer available")

