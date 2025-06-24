# 🔦 KAN-NeRF: TinyNeRF with Kernel-based Activation Networks

This project extends [TinyNeRF-pytorch](https://github.com/volunt4s/TinyNeRF-pytorch) by replacing its MLP networks with **KAN (Kernel-based Activation Networks)**. Two variants are implemented:
- **Position-KAN NeRF**: Only replaces the positional MLP with KAN.
- **Full-KAN NeRF**: Uses KAN for both the density and color networks.

## 📦 Environment Setup

```bash
# Python 3.8+ is recommended
# Install core dependencies
pip install torch numpy matplotlib tqdm imageio scikit-image lpips

# Install efficient-kan (full-featured KAN)
pip install git+https://github.com/Blealtan/efficient-kan.git

# (Optional) Install tinykan if you want the lightweight version
pip install git+https://github.com/Blealtan/tinykan.git
```

## 📂 Dataset Preparation

Download the test dataset used by TinyNeRF:

```bash
wget https://github.com/volunt4s/TinyNeRF-pytorch/raw/main/tiny_nerf_data.npz
```

Then place `tiny_nerf_data.npz` in the root directory of the project.

## 🚀 Run Training

```bash
python tiny_nerf.py
```

This will train the Full-KAN NeRF model and save results in `outputs_full_kan_v2/`.

## 📊 Output

After training, the following results will be saved:
- `metrics.json`: PSNR, SSIM, LPIPS, training time, parameter count, and memory usage.
- `psnr_curve.png`, `ssim_curve.png`, `lpips_curve.png`: Evaluation curves.
- `multi_view.gif`: Rendered views from multiple camera poses.

## 📚 References

- [TinyNeRF-pytorch](https://github.com/volunt4s/TinyNeRF-pytorch)
- [efficient-kan](https://github.com/Blealtan/efficient-kan)
- [tinykan](https://github.com/Blealtan/tinykan)
