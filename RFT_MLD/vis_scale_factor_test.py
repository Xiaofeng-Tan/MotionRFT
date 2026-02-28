#!/usr/bin/env python3
"""
测试 vae_scale_factor 对 VAE decode 结果的影响。

分别使用/不使用 scale_factor 进行 decode，可视化两个 motion，
打印 scale_factor 本身大小，然后退出。
"""
import os
import sys
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from textwrap import wrap

from omegaconf import OmegaConf

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.temos_utils import lengths_to_mask
from reward_adapter import process_T5_outputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# ============ 263 -> joints22 (from visualize_263.py) ============
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

def qinv(q):
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    assert q.shape[-1] == 4 and v.shape[-1] == 3
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num=22):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions

# ============ Visualization (from visualize_263.py) ============
def plot_motion_animation(joints, kinematic_chain, save_path, title="Motion Animation", fps=20, radius=3):
    title = '\n'.join(wrap(title, 40))
    data = joints.copy().reshape(len(joints), -1, 3)
    data *= 1.3
    fig = plt.figure(figsize=(6, 6))
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(False)
        return []

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    def update(index):
        ax.clear()
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0,
                     MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
        for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2],
                      linewidth=4.0 if i < 5 else 2.0, color=color)
        ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2],
                   c='black', s=20, depthshade=True)
        ax.axis('off')
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        return []

    ani = FuncAnimation(fig, update, frames=data.shape[0], interval=1000 / fps,
                        repeat=True, init_func=init, blit=False)
    if save_path.endswith('.gif'):
        ani.save(save_path, writer='pillow', fps=fps)
    else:
        ani.save(save_path, fps=fps)
    print(f"动画已保存到: {save_path}")
    plt.close()


def feats263_to_joints(feats_tensor):
    """(B, T, 263) tensor -> list of (T_i, 22, 3) numpy arrays"""
    joints = recover_from_ric(feats_tensor, joints_num=22)
    return joints.numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============ 1. 加载模型和数据 ============
    cfg = parse_args()
    dataset = get_dataset(cfg)
    train_dataloader = dataset.train_dataloader()

    model = MLD(cfg, dataset)

    assert cfg.TRAIN.PRETRAINED, "cfg.TRAIN.PRETRAINED must not be None."
    logger.info(f"Loading pre-trained model: {cfg.TRAIN.PRETRAINED}")
    state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.vae.eval()
    model.text_encoder.eval()
    model.denoiser.eval()
    model.to(device)

    # ============ 2. 打印 vae_scale_factor ============
    print("\n" + "=" * 60)
    print(f"  vae_scale_factor = {model.vae_scale_factor}")
    print(f"  type: {type(model.vae_scale_factor)}")
    print("=" * 60 + "\n")

    # ============ 3. 获取一个 batch 的数据 ============
    batch = next(iter(train_dataloader))
    from mld.utils.utils import move_batch_to_device
    batch = move_batch_to_device(batch, device)

    texts = batch["text"]
    feats_ref = batch["motion"]
    lengths = batch["length"]
    mask = batch["mask"]

    # 只取第一个样本
    idx = 0
    text = texts[idx]
    length = lengths[idx]
    print(f"Text: {text}")
    print(f"Length: {length}")

    # ============ 4. Diffusion reverse 获得 latents ============
    with torch.no_grad():
        if model.do_classifier_free_guidance:
            texts_cfg = texts + [""] * len(texts)
        else:
            texts_cfg = texts

        t_len, token_embeddings, text_emb = process_T5_outputs(texts_cfg, model.text_encoder.text_model)
        latents = torch.randn((feats_ref.shape[0], *model.latent_dim), device=device)
        latents = model._diffusion_reverse(latents, text_emb, controlnet_cond=None)

    print(f"\nLatents shape: {latents.shape}")
    print(f"Latents range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
    print(f"Latents mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")

    # ============ 5. 分别 decode: 使用/不使用 scale_factor ============
    with torch.no_grad():
        # 使用 scale_factor（正常做法）
        feats_with_scale = model.vae.decode(latents / model.vae_scale_factor, mask)
        # 不使用 scale_factor（直接 decode）
        feats_no_scale = model.vae.decode(latents, mask)

    # 取第一个样本
    feat_with = feats_with_scale[idx, :length].detach().cpu()
    feat_no = feats_no_scale[idx, :length].detach().cpu()

    print(f"\n--- 使用 scale_factor (/ {model.vae_scale_factor}) ---")
    print(f"  feats shape: {feat_with.shape}")
    print(f"  feats range: [{feat_with.min().item():.4f}, {feat_with.max().item():.4f}]")
    print(f"  feats mean: {feat_with.mean().item():.4f}, std: {feat_with.std().item():.4f}")

    print(f"\n--- 不使用 scale_factor ---")
    print(f"  feats shape: {feat_no.shape}")
    print(f"  feats range: [{feat_no.min().item():.4f}, {feat_no.max().item():.4f}]")
    print(f"  feats mean: {feat_no.mean().item():.4f}, std: {feat_no.std().item():.4f}")

    # 对比差异
    diff = (feat_with - feat_no).abs()
    print(f"\n--- 差异 (|with_scale - no_scale|) ---")
    print(f"  max diff: {diff.max().item():.6f}")
    print(f"  mean diff: {diff.mean().item():.6f}")
    print(f"  相同? {torch.allclose(feat_with, feat_no, atol=1e-5)}")

    # ============ 6. 转换为 joints 并可视化 ============
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis_scale_factor_output")
    os.makedirs(output_dir, exist_ok=True)

    # 263 -> joints22
    joints_with = feats263_to_joints(feat_with.unsqueeze(0))  # (1, T, 22, 3)
    joints_no = feats263_to_joints(feat_no.unsqueeze(0))

    joints_with = joints_with[0]  # (T, 22, 3)
    joints_no = joints_no[0]

    print(f"\n--- Joints (with scale) ---")
    print(f"  shape: {joints_with.shape}, range: [{joints_with.min():.4f}, {joints_with.max():.4f}]")
    print(f"\n--- Joints (no scale) ---")
    print(f"  shape: {joints_no.shape}, range: [{joints_no.min():.4f}, {joints_no.max():.4f}]")

    # 可视化
    safe_text = text[:50].replace(" ", "_").replace("/", "_")
    
    path_with = os.path.join(output_dir, f"with_scale_{safe_text}.gif")
    path_no = os.path.join(output_dir, f"no_scale_{safe_text}.gif")

    print(f"\n生成动画中...")
    plot_motion_animation(
        joints_with, T2M_KINEMATIC_CHAIN, path_with,
        title=f"WITH scale (/{model.vae_scale_factor}): {text[:40]}", fps=20
    )
    plot_motion_animation(
        joints_no, T2M_KINEMATIC_CHAIN, path_no,
        title=f"NO scale: {text[:40]}", fps=20
    )

    # ============ 7. 最终总结 ============
    print("\n" + "=" * 60)
    print(f"  vae_scale_factor = {model.vae_scale_factor}")
    print(f"  结果是否相同: {torch.allclose(feat_with, feat_no, atol=1e-5)}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    # 退出
    sys.exit(0)


if __name__ == "__main__":
    main()
