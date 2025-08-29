"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def compute_next_campos(cam_H, action_id, forward_step_size=0.065, turn_angle=10.):
    next_H  = cam_H.copy()

    if action_id == 1:
        # Move forward along z-axis
        next_H[:3, [3]] =  cam_H[:3, [3]] + cam_H[:3, :3] @ np.array([[0.], [0.], [forward_step_size]])
    elif action_id == 2:
        R = cam_H[:3, :3] @ np.array(
            [[np.cos(np.deg2rad(turn_angle)), 0., -np.sin(np.deg2rad(turn_angle))],
             [0., 1., 0.],
             [np.sin(np.deg2rad(turn_angle)), 0, np.cos(np.deg2rad(turn_angle))]]
        )
        next_H[:3, :3] = R
    elif action_id == 3:
        R = cam_H[:3, :3] @ np.array(
            [[np.cos(np.deg2rad(turn_angle)), 0., np.sin(np.deg2rad(turn_angle))],
             [0., 1., 0.],
             [-np.sin(np.deg2rad(turn_angle)), 0, np.cos(np.deg2rad(turn_angle))]]
        )
        next_H[:3, :3] = R
    
    return next_H


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def masked_l1_rgb(x, y, color_mask):
    """
    x,y: (3,H,W)
    color_mask: (3,H,W) con 0/1 (float o bool)
    """
    if color_mask.dtype != torch.float32 and color_mask.dtype != torch.float64:
        color_mask = color_mask.float()
    den = color_mask.sum().clamp_min(1)
    return (torch.abs(x - y) * color_mask).sum() / den


def masked_l1_depth(dpred, dgt, mask):
    """
    dpred,dgt: (1,H,W)
    mask: (1,H,W) con 0/1 (float o bool)
    """
    if mask.dtype != torch.float32 and mask.dtype != torch.float64:
        mask = mask.float()
    den = mask.sum().clamp_min(1)
    return (torch.abs(dpred - dgt) * mask).sum() / den


def calc_ssim_masked(img1, img2, mask, window_size=11):
    """
    img1,img2: (3,H,W)
    mask: (1,H,W) o (H,W) con 0/1 (float o bool)
    Ritorna: SSIM medio pesato dalla maschera (scalare)
    Usa la stessa logica della tua calc_ssim/_ssim, ma senza mediare su tutta l’immagine.
    """
    # porta a (1,C,H,W)
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # porta la mask a (1,1,H,W)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        if mask.shape[0] != 1:
            mask = mask.unsqueeze(0)     # (1,1,H,W) se era (1,H,W)
        else:
            mask = mask.unsqueeze(0)     # (1,1,H,W)

    mask = mask.float()
    channel = img1.size(1)
    window = create_window(window_size, channel).type_as(img1)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    # === copia della tua _ssim per ottenere la MAPPA ===
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12  = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))   # (1,C,H,W)

    # media sui canali → (1,1,H,W)
    ssim_map = ssim_map.mean(dim=1, keepdim=True)

    den = mask.sum().clamp_min(1)
    return (ssim_map * mask).sum() / den


def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points(to_remove, params, variables, optimizer = None):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans']]
    if optimizer is not None:
        for k in keys:
            group = [g for g in optimizer.param_groups if g['name'] == k][0]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state
                params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
                params[k] = group["params"][0]
    else:
        for k in keys:
            params[k] = params[k][to_keep]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    if 'timestep' in variables.keys():
        variables['timestep'] = variables['timestep'][to_keep]
    # NOTE: we have an implicit bug here that shape of varaibles and params are not the same
    # this is from SplaTAM as they didn't enable densification 
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


import torch.nn.functional as F
def get_gaussians_outside_mask(params, curr_data, obj_mask_2d):
    """
    Conta gaussiane dentro/fuori maschera nella vista corrente e stampa
    anche il range delle scale (max asse) per dentro/fuori.

    Ritorna:
      outside_mask: (N,) bool
      stats: dict con conteggi e range scale
    """
    device = params['means3D'].device
    obj_mask_2d = obj_mask_2d.to(device).bool()

    # --- Camera pose corrente (w2c) dal frame corrente ---
    time_idx = curr_data['id']
    cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
    cam_tran = params['cam_trans'][..., time_idx]
    w2c = torch.eye(4, device=device, dtype=torch.float32)
    w2c[:3, :3] = build_rotation(cam_rot)
    w2c[:3, 3]  = cam_tran

    # --- Trasformazione & proiezione ---
    means3D   = params['means3D']                            # (N,3)
    means_cam = (w2c[:3, :3] @ means3D.T + w2c[:3, 3:4]).T   # (N,3)
    z         = means_cam[:, 2]

    K = curr_data['intrinsics'].to(device)                   # (3,3)
    pts_cam_h = torch.cat([means_cam, torch.ones_like(z[:, None])], dim=1)  # (N,4) se serve
    # Proiezione pinhole (senza distorsione)
    pts2d_h = (K @ means_cam.T).T                            # (N,3)
    u = pts2d_h[:, 0] / (pts2d_h[:, 2].clamp_min(1e-6))
    v = pts2d_h[:, 1] / (pts2d_h[:, 2].clamp_min(1e-6))

    H, W = obj_mask_2d.shape[-2], obj_mask_2d.shape[-1]
    in_img = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    iu = u[in_img].round().long().clamp(0, W - 1)
    iv = v[in_img].round().long().clamp(0, H - 1)

    inside_mask_img = torch.zeros(means3D.shape[0], dtype=torch.bool, device=device)
    inside_mask_img[in_img] = obj_mask_2d[iv, iu]  # True se proietta dentro la maschera

    outside_mask = ~inside_mask_img  # tutto ciò che non cade in maschera (incluse fuori immagine o dietro camera)

    # --- Statistiche scale (max asse) ---
    # log_scales: (N,3) -> scale_max: (N,)
    scale_max = torch.exp(params['log_scales']).max(dim=1).values

    idx_in  = torch.nonzero(inside_mask_img, as_tuple=False).squeeze(1)
    idx_out = torch.nonzero(outside_mask,    as_tuple=False).squeeze(1)

    def rng(x):
        if x.numel() == 0:
            return (float('nan'), float('nan'))
        return (x.min().item(), x.max().item())

    in_min,  in_max  = rng(scale_max[idx_in])   # range scale per gaussiane DENTRO maschera
    out_min, out_max = rng(scale_max[idx_out])  # range scale per gaussiane FUORI maschera

    # --- Stampe utili ---
    # print(f"Gaussians  IN  mask: {idx_in.numel()} / {means3D.shape[0]}  | scale_max range: [{in_min:.5f}, {in_max:.5f}]")
    # print(f"Gaussians OUT mask: {idx_out.numel()} / {means3D.shape[0]}  | scale_max range: [{out_min:.5f}, {out_max:.5f}]")

    stats = {
        'in_count':  int(idx_in.numel()),
        'out_count': int(idx_out.numel()),
        'total':     int(means3D.shape[0]),
        'inside_scale_min':  in_min,
        'inside_scale_max':  in_max,
        'outside_scale_min': out_min,
        'outside_scale_max': out_max,
        'idx_in':  idx_in,
        'idx_out': idx_out
    }
    return outside_mask, stats

def prune_gaussians(params, variables, optimizer, iter, prune_dict, scene_bound=None, curr_data=None, obj_mask_2d=None):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()

            # remove points outof bounds
            # if scene_bound is not None:
            #     lower_bound, upper_bound = scene_bound
            #     lower_bound, upper_bound = torch.from_numpy(lower_bound).cuda(), torch.from_numpy(upper_bound).cuda()
            #     to_remove = torch.logical_or(to_remove, torch.logical_or(
            #         torch.any(params['means3D'] < lower_bound - 0.05, dim=1),
            #         torch.any(params['means3D'] > upper_bound + 0.05, dim=1)
            #     ))


            # ADDED OBJECT AWARE MASK PRUNING
            if obj_mask_2d is not None:
                active_alpha_thresh = prune_dict.get('outside_opacity_thresh', 0.01)
                alpha = torch.sigmoid(params['logit_opacities']).squeeze(-1)
                active = (alpha >= active_alpha_thresh)

                # conta/rileva fuori maschera nella vista corrente
                outside_mask, stats = get_gaussians_outside_mask(params, curr_data, obj_mask_2d)
                outside_active = outside_mask & active
                print(f"Outside Mask before pruning: {stats['out_count']}")
                # (opzionale) limita anche per scale troppo grandi
                if 'outside_max_scale' in prune_dict:
                    scale_max = torch.exp(params['log_scales']).max(dim=1).values
                    outside_active = outside_active & (scale_max >= prune_dict['outside_max_scale'])

                num_out = int(outside_active.sum().item())
                if num_out > 0:
                    # debug utile
                    scale_max = torch.exp(params['log_scales']).max(dim=1).values
                    rm_min = float(scale_max[outside_active].min().item())
                    rm_max = float(scale_max[outside_active].max().item())
                    print(f"[Prune] outside-mask: removing {num_out} | scale_max in [{rm_min:.5f}, {rm_max:.5f}]")

                to_remove = torch.logical_or(to_remove, outside_active)

            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)
            torch.cuda.empty_cache()
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
        
        # DAVIDE
        if obj_mask_2d is not None:
            outside_mask, stats = get_gaussians_outside_mask(params, curr_data, obj_mask_2d)
            print(f"Outside Mask after pruning: {stats['out_count']}")
            
    return params, variables


def densify(params, variables, optimizer, iter, densify_dict):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.05))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]
            variables["timestep"] = torch.cat([variables["timestep"], torch.zeros((to_clone.sum(), ), device="cuda")])

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.max(torch.exp(params['log_scales']), dim=1).values > 0.05
            n = densify_dict['num_to_split_into']  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]
            variables["timestep"] = torch.cat([variables["timestep"], torch.zeros((to_split.sum() * n, ), device="cuda")])

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def update_learning_rate(optimizer, means3D_scheduler, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in optimizer.param_groups:
            if param_group["name"] == "means3D":
                lr = means3D_scheduler(iteration)
                param_group['lr'] = lr
                return lr


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper