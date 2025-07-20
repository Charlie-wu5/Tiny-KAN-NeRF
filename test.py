#!/usr/bin/env python3
#先粗判断哪里有东西，做一个box，后续只取穿过box的光线
"""
loss_test_coarse2fine.py  (改良版)
TinyNeRF + Efficient-KAN  |  Hierarchical Sampling (表面附近采样)
"""

import os, time, json, math, random
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_np
import lpips, imageio, matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_loss
from efficient_kan import KAN

# ----------------- 超参 -----------------
OUT_DIR         = "outputs_hierarchical"
PRETRAIN_STEPS  = 800           # ← 预热步数 (保持不变)
N_ITERS_FINE    = 10000
N_COARSE        = 32            # <<< NEW: 粗采样点数
N_FINE          = 32            # <<< NEW: 沿粗采样分布的精细采样点数
PATCH_SIZE      = 1024          # <<< CHANGED: Patch size is now just a ray count
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
L_embed_pos, L_embed_dir = 10, 4

def posenc(L, x):
    rets = [x]
    for i in range(L):
        for fn in (torch.sin, torch.cos):
            rets.append(fn((2.0**i) * x))
    return torch.cat(rets, -1)

class NeRFWithSkip(nn.Module):
    def __init__(self, hidden_dim=128, feature_dim=128):
        super().__init__()
        self.pos_dim = 3 + 3*2*L_embed_pos
        self.dir_dim = 3 + 3*2*L_embed_dir
        self.sigma1 = KAN([self.pos_dim, hidden_dim, hidden_dim])
        self.sigma2 = KAN([hidden_dim + self.pos_dim, hidden_dim, feature_dim + 1])
        self.color  = KAN([self.dir_dim + feature_dim, hidden_dim//2, 3])
    def forward(self, x, d):
        x_enc, d_enc = posenc(L_embed_pos, x), posenc(L_embed_dir, d)
        h1 = self.sigma1(x_enc)
        h2 = self.sigma2(torch.cat([h1, x_enc], -1))
        sigma, feat = F.relu(h2[...,0]), h2[...,1:]
        rgb = torch.sigmoid(self.color(torch.cat([feat, d_enc], -1)))
        return rgb, sigma

class CoarseNet(NeRFWithSkip):
    def __init__(self):
        super().__init__(hidden_dim=64, feature_dim=64)

def get_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(torch.arange(W,device=device),
                          torch.arange(H,device=device), indexing='xy')
    dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[...,None,:]*c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

# ----- Render function used for single-pass rendering (e.g. pre-training) -----
def render_rays(model, rays_o, rays_d, near, far, N_samples, rand=False, chunk=4096):
    device = rays_o.device
    z_vals = torch.linspace(near, far, N_samples, device=device)
    if rand:
        z_vals = z_vals + torch.rand(rays_o.shape[0],1,device=device)*(far-near)/N_samples
    else:
        z_vals = z_vals.expand([rays_o.shape[0], N_samples])
    pts  = rays_o[...,None,:] + rays_d[...,None,:]*z_vals[..., :,None]
    dirs = rays_d[...,None,:].expand_as(pts)
    pts_f, dirs_f = pts.reshape(-1,3), dirs.reshape(-1,3)

    rgb_all, sig_all = [], []
    for i in range(0, pts_f.shape[0], chunk):
        r, s = model(pts_f[i:i+chunk], dirs_f[i:i+chunk])
        rgb_all.append(r); sig_all.append(s)
    rgb   = torch.cat(rgb_all,0).view(-1,N_samples,3)
    sigma = torch.cat(sig_all,0).view(-1,N_samples)

    dists = torch.cat([z_vals[...,1:]-z_vals[...,:-1],
                       torch.full(z_vals[...,:1].shape,1e10,device=device)], -1)
    alpha  = 1 - torch.exp(-sigma * dists)
    T      = torch.cumprod(torch.cat([torch.ones(alpha.shape[0],1,device=device),
                                      1-alpha+1e-10], -1), -1)[:,:-1]
    weights = alpha * T
    rgb_map = torch.sum(weights[...,None]*rgb, -2)
    depth   = torch.sum(weights*z_vals, -1)
    return rgb_map, depth, weights, z_vals


# <<< MODIFIED FUNCTION (FIXED): Hierarchical rendering >>>
def render_rays_hierarchical(coarse_model, fine_model, rays_o, rays_d, near, far,
                             N_coarse, N_fine, rand=False, chunk=4096):
    # 1. Coarse Pass: Get weights from the coarse model
    with torch.no_grad():
        _, _, weights_coarse, z_vals_coarse = render_rays(
            coarse_model, rays_o, rays_d, near, far, N_coarse, rand=rand, chunk=chunk
        )

    # 2. Importance Sampling: Sample new points based on coarse weights
    z_vals_mid = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
    weights = weights_coarse[..., 1:-1] + 1e-5 # Add epsilon for stability
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Inverse transform sampling
    if rand:
        u = torch.rand(list(cdf.shape[:-1]) + [N_fine], device=cdf.device)
    else:
        u = torch.linspace(0., 1., steps=N_fine, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_fine])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(z_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_samples_fine = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    # 3. Fine Pass: Render with combined sample points
    # Combine coarse and fine samples and sort them
    z_vals_all, _ = torch.sort(torch.cat([z_vals_coarse, z_samples_fine], -1), -1)
    
    # <<< CORE FIX START >>>
    # Use the combined z_vals_all to render with the fine model.
    # This logic is copied from render_rays, but uses z_vals_all directly.
    
    # Calculate points in 3D space using the combined samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_all[..., :, None]
    dirs = rays_d[..., None, :].expand_as(pts)
    pts_f, dirs_f = pts.reshape(-1, 3), dirs.reshape(-1, 3)

    # Query the fine model
    rgb_all, sig_all = [], []
    for i in range(0, pts_f.shape[0], chunk):
        r, s = fine_model(pts_f[i:i+chunk], dirs_f[i:i+chunk])
        rgb_all.append(r)
        sig_all.append(s)
    
    N_samples_total = N_coarse + N_fine
    rgb = torch.cat(rgb_all, 0).view(-1, N_samples_total, 3)
    sigma = torch.cat(sig_all, 0).view(-1, N_samples_total)

    # Perform volumetric rendering using the combined samples
    dists = z_vals_all[..., 1:] - z_vals_all[..., :-1]
    dists = torch.cat([dists, torch.full(z_vals_all[..., :1].shape, 1e10, device=dists.device)], -1)
    
    alpha = 1. - torch.exp(-sigma * dists)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1, device=alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T
    
    rgb_map_fine = torch.sum(weights[..., None] * rgb, -2)
    depth_fine = torch.sum(weights * z_vals_all, -1)
    
    # <<< CORE FIX END >>>

    return rgb_map_fine, depth_fine, None, None # Return only final results


def main():
    # <<< NEW Hyperparameter for rendering >>>
    RENDER_CHUNK_SIZE = 1024 # Adjust this based on your GPU memory. Lower if you still get OOM errors.

    device = torch.device(DEVICE); torch.set_default_dtype(torch.float32)
    data   = np.load('tiny_nerf_data.npz')
    images = torch.from_numpy(data['images'][..., :3])
    poses  = torch.from_numpy(data['poses'])
    focal  = float(data['focal'])
    H, W   = images.shape[1:3]

    # ---------- Stage 0 预热粗网 ----------
    coarse_net = CoarseNet().to(device)
    opt_c = torch.optim.Adam(coarse_net.parameters(), 5e-4)
    for step in tqdm(range(PRETRAIN_STEPS), desc="Pretrain coarse"):
        coarse_net.train()
        idx  = np.random.randint(0,100)
        img  = images[idx].to(device)
        c2w  = poses[idx].to(device)
        ro, rd = get_rays(H,W,focal,c2w,device)
        sel = torch.randint(0,H*W,(PATCH_SIZE,),device=device)
        rgb_gt = img.view(-1,3)[sel]
        rgb_pr,_,_,_ = render_rays(coarse_net,
                                  ro.view(-1,3)[sel], rd.view(-1,3)[sel],
                                  2.,6., N_COARSE, rand=True)
        loss = F.mse_loss(rgb_pr, rgb_gt)
        opt_c.zero_grad(); loss.backward(); opt_c.step()

    # ---------- Stage 1 Fine 训练 ----------
    finenet = NeRFWithSkip().to(device)
    optim_f = torch.optim.Adam(finenet.parameters(), 1e-3)
    sched_f = torch.optim.lr_scheduler.LambdaLR(
        optim_f, lambda it: 0.5*(1+math.cos(math.pi*it/N_ITERS_FINE)) )
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    coarse_net.eval()
    for p in coarse_net.parameters():
        p.requires_grad = False

    psnr_l, ssim_l, lpips_l, iters_l, loss_l = [],[],[],[],[]
    for it in tqdm(range(N_ITERS_FINE), desc="Fine train"):
        finenet.train()
        
        idx = np.random.randint(0,100)
        img = images[idx].to(device)
        c2w = poses[idx].to(device)
        ro_img, rd_img = get_rays(H,W,focal,c2w,device)
        sel = torch.randint(0, H*W, (PATCH_SIZE,), device=device)
        ro, rd = ro_img.view(-1,3)[sel], rd_img.view(-1,3)[sel]
        rgb_gt = img.view(-1,3)[sel]
        
        rgb_pr, _, _, _ = render_rays_hierarchical(
            coarse_net, finenet, ro, rd, 2., 6., N_COARSE, N_FINE, rand=True
        )
        loss = F.mse_loss(rgb_pr, rgb_gt)
        optim_f.zero_grad(); loss.backward(); optim_f.step(); sched_f.step()
        loss_l.append(loss.item())

        if (it+1)%500==0 or it==0:
            finenet.eval()
            with torch.no_grad():
                test_idx = 101
                img_t = images[test_idx].to(device)
                pose_t= poses[test_idx].to(device)
                ro_t_all, rd_t_all = get_rays(H,W,focal,pose_t,device)
                
                # Evaluation also needs to be chunked to avoid OOM
                ro_flat = ro_t_all.view(-1, 3)
                rd_flat = rd_t_all.view(-1, 3)
                rgb_chunks = []
                for i in range(0, ro_flat.shape[0], RENDER_CHUNK_SIZE):
                    rgb_chunk,_,_,_ = render_rays_hierarchical(
                        coarse_net, finenet, 
                        ro_flat[i:i+RENDER_CHUNK_SIZE], rd_flat[i:i+RENDER_CHUNK_SIZE],
                        2., 6., N_COARSE, N_FINE, rand=False
                    )
                    rgb_chunks.append(rgb_chunk)
                
                rgb_t = torch.cat(rgb_chunks, 0).view(H,W,3)
                
                mse = F.mse_loss(rgb_t, img_t)
                psnr = float((-10*torch.log10(mse)).item())
                ssim_v = float(ssim_np(rgb_t.cpu().numpy(), img_t.cpu().numpy(),
                                       data_range=1.0, channel_axis=2))
                lpips_v = float(lpips_fn((rgb_t.permute(2,0,1)[None]*2-1),
                                        (img_t.permute(2,0,1)[None]*2-1)).item())
                psnr_l.append(psnr); ssim_l.append(ssim_v); lpips_l.append(lpips_v)
                iters_l.append(it+1)
                print(f"[{it+1}] PSNR {psnr:.2f} SSIM {ssim_v:.3f} LPIPS {lpips_v:.3f}")

    # ---------- 保存 ----------
    metrics = {"psnr":psnr_l,"ssim":ssim_l,"lpips":lpips_l,
               "iters":iters_l,"loss":loss_l}
    with open(os.path.join(OUT_DIR,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    for name,vals in [("psnr",psnr_l),("ssim",ssim_l),("lpips",lpips_l)]:
        plt.figure(); plt.plot(iters_l,vals,'-o'); plt.title(name.upper())
        plt.savefig(os.path.join(OUT_DIR,f"{name}_curve.png")); plt.close()
    plt.figure(); plt.plot(range(len(loss_l)),loss_l); plt.title("Loss")
    plt.savefig(os.path.join(OUT_DIR,"loss.png")); plt.close()

    # <<< MODIFIED: GIF Rendering with Chunking to prevent OOM error >>>
    finenet.eval()
    frames=[]
    for vid in tqdm(range(0,100,5), desc="Rendering GIF"):
        c2w = poses[vid].to(device)
        ro_all, rd_all = get_rays(H, W, focal, c2w, device)
        
        # Flatten rays for easy chunking
        ro_flat = ro_all.view(-1, 3)
        rd_flat = rd_all.view(-1, 3)
        
        rgb_chunks = []
        # Process rays in manageable chunks
        for i in range(0, ro_flat.shape[0], RENDER_CHUNK_SIZE):
            ro_chunk = ro_flat[i:i+RENDER_CHUNK_SIZE]
            rd_chunk = rd_flat[i:i+RENDER_CHUNK_SIZE]
            
            # Render one chunk of rays
            rgb_chunk, _, _, _ = render_rays_hierarchical(
                coarse_net, finenet, ro_chunk, rd_chunk, 2., 6.,
                N_COARSE, N_FINE, rand=False
            )
            rgb_chunks.append(rgb_chunk)
            
        # Concatenate all rendered chunks into a single image
        full_rgb = torch.cat(rgb_chunks, 0)
        
        # Reshape to image dimensions and append to frame list
        frame_np = (full_rgb.view(H,W,3).cpu().numpy()*255).astype(np.uint8)
        frames.append(frame_np)

    imageio.mimsave(os.path.join(OUT_DIR,"multi_view.gif"), frames, fps=8)
    print("Training done, outputs saved to", OUT_DIR)

if __name__ == "__main__":
    main()