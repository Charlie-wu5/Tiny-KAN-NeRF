#!/usr/bin/env python3
"""
TinyNeRF with efficient-kan KAN Network (Corrected Full KAN Architecture)
增加：PSNR/SSIM/LPIPS、参数量、训练时间、显存、多视角 GIF
依赖：
  pip install torch numpy matplotlib tqdm imageio scikit-image lpips
  pip install git+https://github.com/Blealtan/efficient-kan.git
"""
import os, time, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import imageio
from skimage.metrics import structural_similarity as ssim
import lpips

from efficient_kan import KAN

# 输出路径
out_dir = "outputs_full_kan_v2"
os.makedirs(out_dir, exist_ok=True)

# ===== 1. 定义两个独立的位置编码 =====
L_embed_pos = 10  # 对位置使用更高频率的编码
L_embed_dir = 4  # 对方向使用较低频率的编码


def posenc(L, x):
    rets = [x]
    for i in range(L):
        for fn in (torch.sin, torch.cos):
            rets.append(fn((2.0 ** i) * x))
    return torch.cat(rets, -1)


# ===== 2. 修正NeRF模型架构 =====
class NeRF(nn.Module):
    def __init__(self, hidden_dim=256, feature_dim=256):
        super().__init__()
        # 编码后的维度
        self.pos_dim = 3 + 3 * 2 * L_embed_pos
        self.dir_dim = 3 + 3 * 2 * L_embed_dir

        # 密度网络：输入位置编码，输出sigma和特征向量
        self.sigma_net = KAN([self.pos_dim, hidden_dim, hidden_dim, feature_dim + 1])

        # 颜色网络：输入特征向量和方向编码，输出RGB
        self.color_net = KAN([self.dir_dim + feature_dim, hidden_dim // 2, 3])

    def forward(self, x, d):
        # 对位置和方向进行编码
        x_enc = posenc(L_embed_pos, x)
        d_enc = posenc(L_embed_dir, d)

        # 通过密度网络
        h = self.sigma_net(x_enc)
        sigma = F.relu(h[..., 0])
        feature = h[..., 1:]

        # 将特征向量和方向编码结合，输入到颜色网络
        color_input = torch.cat([feature, d_enc], dim=-1)
        rgb = torch.sigmoid(self.color_net(color_input))

        return rgb, sigma


def get_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(torch.arange(W, device=device, dtype=torch.float32),
                          torch.arange(H, device=device, dtype=torch.float32), indexing="xy")
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# ===== 3. 修正渲染函数以传递方向 =====
def render_rays(model, rays_o, rays_d, near, far, N_samples, rand=False, chunk=1024):
    device = rays_o.device

    def batch_forward(x_pts, d_dirs):
        # 批处理以防OOM
        all_rgb, all_sigma = [], []
        for i in range(0, x_pts.shape[0], chunk):
            rgb_chunk, sigma_chunk = model(x_pts[i:i + chunk], d_dirs[i:i + chunk])
            all_rgb.append(rgb_chunk)
            all_sigma.append(sigma_chunk)
        return torch.cat(all_rgb, 0), torch.cat(all_sigma, 0)

    # 准备采样点
    z_vals = torch.linspace(near, far, N_samples, device=device)
    if rand:
        z_vals = z_vals + torch.rand(rays_o.shape[0], 1, device=device) * (far - near) / N_samples
    else:
        z_vals = z_vals.expand([rays_o.shape[0], N_samples])

    # 计算3D点和对应的方向
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    # (num_rays, N_samples, 3)
    dirs = rays_d[..., None, :].expand_as(pts)
    # (num_rays, N_samples, 3)

    # 将点和方向展平以进行批处理
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)

    # 前向传播
    rgb_flat, sigma_flat = batch_forward(pts_flat, dirs_flat)

    # 恢复形状
    rgb = rgb_flat.view(-1, N_samples, 3)
    sigma = sigma_flat.view(-1, N_samples)

    # 体积渲染
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1],
                       torch.full(z_vals[..., :1].shape, 1e10, device=device)], dim=-1)

    alpha = 1. - torch.exp(-sigma * dists)
    # T_i = exp(-sum_{j<i} sigma_j * delta_j)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1, device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T

    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)

    return rgb_map, depth_map


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    data = np.load('tiny_nerf_data.npz')
    images = torch.from_numpy(data['images'][..., :3])
    poses = torch.from_numpy(data['poses'])
    focal = float(data['focal'])
    H, W = images.shape[1:3]

    test_idx = 101
    model = NeRF(hidden_dim=256, feature_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    N_iters, N_rand, N_samples = 10000, 1024, 32
    eval_every = 500
    psnr_list, ssim_list, lpips_list, eval_iters = [], [], [], []
    start_time = time.time()

    for i in tqdm(range(N_iters)):
        model.train()
        idx = np.random.randint(0, 100)
        target_img = images[idx].to(device)
        c2w = poses[idx].to(device)
        rays_o, rays_d = get_rays(H, W, focal, c2w, device)

        # 从全图中随机采样射线
        coords = torch.stack(
            torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'), -1)
        coords = coords.reshape(-1, 2)
        select_inds = torch.randperm(coords.shape[0])[:N_rand]

        select_coords = coords[select_inds]
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        target_s = target_img[select_coords[:, 0], select_coords[:, 1]]

        rgb_pred, _ = render_rays(model, rays_o, rays_d, 2.0, 6.0, N_samples, rand=True)

        loss = F.mse_loss(rgb_pred, target_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                test_img = images[test_idx].to(device)
                test_pose = poses[test_idx].to(device)
                ro_t, rd_t = get_rays(H, W, focal, test_pose, device)

                # 整图渲染
                rgb_t, _ = render_rays(model, ro_t.reshape(-1, 3), rd_t.reshape(-1, 3), 2.0, 6.0, N_samples, rand=False)

                rgb_t = rgb_t.reshape(H, W, 3)
                mse = F.mse_loss(rgb_t, test_img)
                psnr = float((-10 * torch.log10(mse)).item())

                img_pred_np = rgb_t.cpu().numpy()
                img_gt_np = test_img.cpu().numpy()

                ssim_v = float(ssim(img_pred_np, img_gt_np, data_range=1.0, channel_axis=2))

                # LPIPS需要 (1,3,H,W) BGR [-1,1] 格式
                img_pred_t = torch.from_numpy(img_pred_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
                img_gt_t = torch.from_numpy(img_gt_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
                lpips_v = float(lpips_fn(img_pred_t, img_gt_t).item())

                eval_iters.append(i)
                psnr_list.append(psnr)
                ssim_list.append(ssim_v)
                lpips_list.append(lpips_v)
                tqdm.write(f"[EVAL] Iter {i + 1}: PSNR={psnr:.2f} SSIM={ssim_v:.3f} LPIPS={lpips_v:.3f}")

    # 指标保存、绘图、生成GIF（这部分代码无需修改，故省略）
    # ...

    # 计算参数量、显存等指标
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_alloc = torch.cuda.max_memory_allocated(device) / 1e6
    time_used = time.time() - start_time
    metrics = {
        "params_M": float(param_count / 1e6),
        "train_time_s": float(time_used),
        "max_gpu_mem_MB": float(mem_alloc),
        "psnr": psnr_list,
        "ssim": ssim_list,
        "lpips": lpips_list,
        "eval_iters": eval_iters
    }

    # 保存指标
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 曲线保存
    for name, vals in [("psnr", psnr_list), ("ssim", ssim_list), ("lpips", lpips_list)]:
        plt.figure()
        plt.plot(eval_iters, vals, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel(name.upper())
        plt.title(f"{name.upper()} vs Iteration")
        plt.savefig(os.path.join(out_dir, f"{name}_curve.png"))
        plt.close()

    # 多视角 GIF
    model.eval();
    frames = []
    for vid in range(0, 100, 10):
        c2w = poses[vid].to(device)
        ro_v, rd_v = get_rays(H, W, focal, c2w, device)
        with torch.no_grad():
            rgb_v, _ = render_rays(model, ro_v.reshape(-1, 3), rd_v.reshape(-1, 3), 2.0, 6.0, N_samples)
        img = (rgb_v.reshape(H, W, 3).cpu().numpy() * 255).astype(np.uint8)
        frames.append(img)
    imageio.mimsave(os.path.join(out_dir, "multi_view.gif"), frames, fps=5)


if __name__ == "__main__":
    main()
