#!/usr/bin/env python3
"""
TinyNeRF with efficient-kan KAN Network using DistributedDataParallel (DDP)
依赖:
  pip install torch numpy matplotlib tqdm imageio
  pip install git+https://github.com/Blealtan/efficient-kan.git
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import imageio

# ===== DDP Imports =====
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
# =====================

# 引入 efficient-kan 的 KAN 模块
from efficient_kan import KAN

# 输出目录
out_dir = "outputs_ddp"
# DDP: only main process should create directory
# os.makedirs(out_dir, exist_ok=True)

# 设备与默认精度
# device will be set per process in DDP
torch.set_default_dtype(torch.float32)

# 位置编码
L_embed = 6
def posenc(x):
    rets = [x]
    for i in range(L_embed):
        for fn in (torch.sin, torch.cos):
            rets.append(fn((2.0 ** i) * x))
    return torch.cat(rets, -1)
embed_fn = posenc

# NeRF 网络（使用 efficient-kan 的 KAN）
class NeRF(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.in_dim = 3 + 3 * 2 * L_embed  # ==39
        self.kan = KAN([self.in_dim, hidden_dim, hidden_dim, 4], base_activation=nn.ReLU)

    def forward(self, x, update_grid=False):
        h = embed_fn(x)
        # In DDP, update_grid should be handled carefully if it modifies state
        # For efficient-kan, it seems okay as it's a forward pass argument.
        raw = self.kan(h, update_grid=update_grid)
        return raw

# 生成射线 (device as argument)
def get_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

# 渲染射线 (device as argument)
def render_rays(model, rays_o, rays_d, near, far, N_samples, rand=False, chunk=65536, iter_idx=0):
    device = rays_o.device
    def batchify(fn):
        return lambda x: torch.cat([fn(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], 0)

    def exclusive_cumprod(x):
        cp = torch.cumprod(x, -1)
        cp = torch.roll(cp, 1, -1)
        cp[..., 0] = 1.
        return cp

    rays_shape = rays_o.shape
    z_vals = torch.linspace(near, far, N_samples, device=device)
    if rand:
        z_vals = z_vals + torch.rand(rays_o.shape[0], 1, device=device) * (far - near) / N_samples
    else:
        z_vals = z_vals.expand(rays_shape[0], N_samples)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flat = pts.reshape(-1, 3)
    
    # In DDP, model might need its module attribute accessed if not in forward
    raw = batchify(lambda x: model(x, update_grid=(iter_idx % 50 == 0)))(pts_flat)
    
    # ===== 代码修改处 (健壮性) =====
    # 使用 rays_shape 来恢复形状，这是正确的修复
    # raw = raw.view(rays_shape[0], N_samples, 4) -> This was the bug for H,W inputs
    raw = raw.view(*rays_shape[:-1], N_samples, 4)
    # ============================

    sigma_a = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    diffs = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([diffs, torch.tensor([1e10], device=device).expand(diffs[..., :1].shape)], -1)

    alpha = 1 - torch.exp(-sigma_a * dists)
    weights = alpha * exclusive_cumprod(1 - alpha + 1e-10)

    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)

    if len(rays_shape) == 3:  # H, W, 3
        rgb_map = rgb_map.view(rays_shape[0], rays_shape[1], 3)
        depth_map = depth_map.view(rays_shape[0], rays_shape[1])

    return rgb_map, depth_map

# ===== DDP setup functions =====
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
# =============================

def main_worker(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        os.makedirs(args['out_dir'], exist_ok=True)

    # 加载数据
    data = np.load('tiny_nerf_data.npz')
    images = torch.from_numpy(data['images'][...,:3]) # Move to device later
    poses = torch.from_numpy(data['poses'])
    focal = float(data['focal'])
    H, W = images.shape[1:3]

    # 划分训练/测试 (indices are enough)
    test_idx = 101
    train_indices = list(range(100))

    # DDP: Use a sampler to distribute training data
    train_dataset = TensorDataset(images[:100], poses[:100])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # We will use the sampler to shuffle indices each epoch

    # 模型与优化器
    model = NeRF(hidden_dim=args['hidden_dim']).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    grad_clip_val = 0.1

    # Main training loop
    if rank == 0:
        pbar = tqdm(range(args['N_iters']))
    else:
        pbar = range(args['N_iters'])

    for i in pbar:
        model.train()
        train_sampler.set_epoch(i) # Ensure proper shuffling
        
        # Manually get a random index for this rank
        idx = np.random.randint(0, len(train_indices))
        target_img = images[idx].to(device)
        c2w = poses[idx].to(device)
        
        rays_o, rays_d = get_rays(H, W, focal, c2w, device)
        ro = rays_o.reshape(-1, 3)
        rd = rays_d.reshape(-1, 3)
        select = torch.randperm(ro.shape[0], device=device)[:args['N_rand']]

        rgb_pred, _ = render_rays(model, ro[select], rd[select], 2.0, 6.0, args['N_samples'],
                                  rand=True, iter_idx=i, chunk=args['N_rand'])
        target_pix = target_img.reshape(-1, 3)[select]
        
        loss = F.mse_loss(rgb_pred, target_pix)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        optimizer.step()
        scheduler.step()

        # Evaluation and logging only on rank 0
        if rank == 0 and i % 200 == 0:
            model.eval()
            with torch.no_grad():
                test_img = images[test_idx].to(device)
                test_pose = poses[test_idx].to(device)
                ro_t, rd_t = get_rays(H, W, focal, test_pose, device)
                rgb_t, _ = render_rays(model, ro_t, rd_t, 2.0, 6.0, args['N_samples'],
                                       rand=False, chunk=4096)
                mse = F.mse_loss(rgb_t, test_img)
                if mse > 0:
                    psnr = -10 * torch.log10(mse)
                    pbar.set_description(f"PSNR: {psnr.item():.2f}")
                    # You would save psnr values here
    
    # Cleanup and save final assets from rank 0
    if rank == 0:
        print("Training finished. Rendering GIF...")
        frames = []
        model.eval()
        save_views = list(range(0, 100, 10))
        for vid in tqdm(save_views, desc="Rendering GIF"):
            c2w = poses[vid].to(device)
            ro_v, rd_v = get_rays(H, W, focal, c2w, device)
            with torch.no_grad():
                rgb_v, _ = render_rays(model, ro_v, rd_v, 2.0, 6.0, args['N_samples'],rand=False, chunk=4096)
            img = (rgb_v.cpu().numpy() * 255).astype('uint8')
            frames.append(img)

        gif_path = os.path.join(args['out_dir'], 'multi_view.gif')
        imageio.mimsave(gif_path, frames, fps=8)
        print(f"已保存多视角动画到：{gif_path}")
        
        # Save model
        torch.save(model.module.state_dict(), os.path.join(args['out_dir'], 'model.pth'))


    cleanup()

if __name__ == '__main__':
    # Hyperparameters
    args = {
        'N_samples': 32,
        'N_iters': 10000,
        'N_rand': 512,
        'lr': 5e-4,
        'hidden_dim': 256,
        'out_dir': out_dir,
    }

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Using {world_size} GPUs for DDP training.")
        mp.spawn(main_worker,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        print("Only one GPU found, running in single-GPU mode.")
        # Create a dummy rank 0 process for single GPU
        main_worker(0, 1, args)