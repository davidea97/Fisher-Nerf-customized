import argparse
import logging 
import shlex
from typing import List, Dict, Iterable, Tuple
import torch
from models.SLAM.utils.rot import matrix_to_quaternion
import numpy as np

logger = logging.getLogger("rich")

try:
    from droid_slam import Droid
except Exception as e:
    logger.error("Please install droid slam following the direction in README.md")
    print(e)

class DroidWrapper():
    def __init__(self, image_size: List[int], ckpt_path: str = "./ckpt/droid.pth"):
        parser = argparse.ArgumentParser()
        # parser.add_argument("--imagedir", type=str, help="path to image directory")
        # parser.add_argument("--calib", type=str, help="path to calibration file")
        parser.add_argument("--t0", default=0, type=int, help="starting frame")
        parser.add_argument("--stride", default=3, type=int, help="frame stride")

        parser.add_argument("--stereo", action='store_true')
        parser.add_argument("--weights", default="droid.pth")
        parser.add_argument("--buffer", type=int, default=512)
        parser.add_argument("--image_size", default=[240, 320])
        parser.add_argument("--disable_vis", action="store_true")

        parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
        parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
        parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
        parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
        parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
        parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
        parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
        parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

        parser.add_argument("--backend_thresh", type=float, default=22.0)
        parser.add_argument("--backend_radius", type=int, default=2)
        parser.add_argument("--backend_nms", type=int, default=3)
        parser.add_argument("--upsample", action="store_true")
        parser.add_argument("--reconstruction_path", help="path to saved reconstruction")

        args = parser.parse_args(shlex.split(f"--weights {ckpt_path} --disable_vis"))

        args.image_size = image_size 
        self.droid = Droid(args)
    
    def track(self, tstamp, image, depth=None, intrinsics=None):
        self.droid.track(tstamp, image, depth, intrinsics)
    
    def terminate(self, stream: Iterable):
        self.droid.terminate(stream)
    
    def est_pose(self, keyframe_list: Iterable[Dict], tstamp: int, image, depth=None, intrinsics: torch.Tensor = None):
        torch.cuda.empty_cache()
        self.droid.backend(7)

        torch.cuda.empty_cache()
        self.droid.backend(12)

        def stream():
            for keyframe in keyframe_list:
                yield keyframe['id'], keyframe['droid_color'], intrinsics
            yield tstamp, image, intrinsics

        camera_trajectory = self.droid.traj_filler(stream()).detach()
        
        def SE3_to_quant_trans(G) -> Tuple[torch.Tensor, torch.Tensor]:
            mat = G.matrix()
            quat = matrix_to_quaternion(mat[:3, :3])
            trans = mat[:3, 3]
            return quat, trans

        torch.cuda.empty_cache()
        return list(map(SE3_to_quant_trans, camera_trajectory))