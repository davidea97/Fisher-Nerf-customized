
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
from tester_navigator import Navigator

def nav_testing(options, scene_id, dynamic_scene=False, dino_extraction=False, save_data=False):
    navigator = Navigator(options, scene_id, dynamic_scene, dino_extraction, save_data)
    navigator.frontier_test_navigation()

if __name__ == '__main__':
    __spec__ = None
    options = TrainOptions().parse_args()

    options.dataset_type = "hm3d"
    options.split = "val"
    options.img_size = 256
    scene_ids = options.scenes_list
    dynamic_scene = False
    dino_extraction = False
    save_data = False
    
    # Create iterables for map function
    n = len(scene_ids)
    options_list = [options] * n
    args = [*zip(options_list, scene_ids)]
    nav_testing(*args[0], dynamic_scene, dino_extraction, save_data)
    
    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")

