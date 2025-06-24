# TinyNeRF with KAN-based Networks

This project extends [volunt4s/TinyNeRF-pytorch](https://github.com/volunt4s/TinyNeRF-pytorch) by integrating **Kernel Attention Networks (KAN)** into the NeRF architecture.

## Implemented Models

- ✅ **Positional-KAN NeRF**: Only replaces the MLP for positional encoding with a KAN layer.
- ✅ **Full-KAN NeRF**: Fully replaces both the density and color MLPs with KAN layers using `efficient-kan`.

## Features

- Efficient rendering with KAN-based networks
- Evaluation with **PSNR**, **SSIM**, and **LPIPS**
- GPU memory & training time logging
- Animated multi-view GIF rendering

## Installation

Install Python dependencies:

```bash
pip install torch numpy matplotlib tqdm imageio scikit-image lpips
