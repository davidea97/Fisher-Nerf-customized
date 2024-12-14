import cv2
import numpy as np
import argparse
import os
import os.path as osp

arg = argparse.ArgumentParser()
arg.add_argument('--exp_dir', type=str, default=None)

opt = arg.parse_args()

tracking_dir = osp.join(opt.exp_dir, "tracking")
tracking_files = os.listdir(tracking_dir)

os.makedirs(osp.join(opt.exp_dir, "video"), exist_ok=True)

# get the ids
ids = list(map(lambda x: int(x.split(".")[0]), tracking_files))

for idx in ids:
    pathviz_file = osp.join(opt.exp_dir, "pathviz", f"gt_pos_path_{idx}.png")
    tracking_file = osp.join(opt.exp_dir, "tracking", "{:04d}.png".format(idx))
    gaussian_bev = osp.join(opt.exp_dir, "gaussian_bev", f"iteration_step_{idx}.png")

    pathviz = cv2.imread(pathviz_file)
    tracking = cv2.imread(tracking_file)
    gaussian = cv2.imread(gaussian_bev)

    if (pathviz is None) or (tracking is None) or (gaussian is None):
        print(f"Skipping {idx}")
        continue

    # canvas is 1200 x 1200
    canvas = np.ones((800, 1400, 3), dtype=np.uint8) * 255

    tracking_patch = tracking[340:340+245, 8:8+540]
    
    # paint tracking (centralized)
    canvas[ 80: 80+245, 30:30+540] = tracking_patch

    # paint pathviz (lower left)
    pathviz = cv2.resize(pathviz, (540, 378))
    # pathviz = np.transpose(pathviz, (1, 0, 2))
    canvas[411:411 + 378, 30:30+540] = pathviz

    # gaussian = np.transpose(gaussian, (1, 0, 2))
    canvas[:, 600:600 + 800] = gaussian

    cv2.imwrite(osp.join(opt.exp_dir, "video", "{:04d}.png".format(idx)), canvas)
