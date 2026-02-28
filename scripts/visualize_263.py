#!/usr/bin/env python3
"""
可视化 HumanML3D 263 维特征的 .npy 文件
数据格式: (num_frames, 263)

263维特征组成:
- [0]:   root rotation velocity (1)
- [1:3]: root linear velocity on xz plane (2)
- [3]:   root height y (1)
- [4:67]: local joint positions relative to root, 21 joints × 3 (63)
- [67:193]: joint velocities (63) + rotations (63)
- [193:259]: foot contact etc.
- ...

先通过 recover_from_ric 转换为 (frames, 22, 3) 再绘制火柴人。
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import os
from textwrap import wrap


T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


# ============ 263 -> joints22 ============
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
    """263维特征 -> (seq_len, joints_num, 3)"""
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


# ============ Visualization ============
def plot_single_frame(joints, kinematic_chain, save_path=None, title="Motion Frame"):
    if len(joints.shape) == 3:
        joints = joints[0]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='black', s=30, depthshade=True)
    for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
        ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2],
                  linewidth=3.0 if i < 3 else 2.0, color=color)
    for idx in range(joints.shape[0]):
        ax.text(joints[idx, 0], joints[idx, 1], joints[idx, 2], str(idx), fontsize=8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    max_range = np.array([joints[:, d].max() - joints[:, d].min() for d in range(3)]).max() / 2.0
    mid = [(joints[:, d].max() + joints[:, d].min()) * 0.5 for d in range(3)]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"单帧图像已保存到: {save_path}")
    plt.close()


def plot_motion_animation(joints, kinematic_chain, save_path, title="Motion Animation", fps=20, radius=3):
    title = '\n'.join(wrap(title, 30))
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


def plot_motion_sequence(joints, kinematic_chain, save_path, num_frames=8, title="Motion Sequence"):
    total_frames = joints.shape[0]
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    fig, axes = plt.subplots(2, num_frames // 2, figsize=(4 * (num_frames // 2), 8),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    for ax_idx, frame_idx in enumerate(frame_indices):
        ax = axes[ax_idx]
        fj = joints[frame_idx]
        ax.scatter(fj[:, 0], fj[:, 1], fj[:, 2], c='black', s=20)
        for chain, color in zip(kinematic_chain, colors):
            ax.plot3D(fj[chain, 0], fj[chain, 1], fj[chain, 2], linewidth=2, color=color)
        ax.set_title(f"Frame {frame_idx}", fontsize=10)
        ax.view_init(elev=20, azim=-60)
        ax.axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"帧序列图像已保存到: {save_path}")
    plt.close()


# ============ Main ============
def main():
    parser = argparse.ArgumentParser(description='Visualize HumanML3D 263-dim .npy files')
    parser.add_argument('--npy_path', type=str, required=True,
                        help='Path to .npy file with shape (frames, 263)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'visualizations', 'humanml3d_263'))
    parser.add_argument('--mode', type=str, default='all',
                        choices=['single', 'animation', 'sequence', 'all'])
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--frame', type=int, default=0)
    args = parser.parse_args()

    print(f"加载数据: {args.npy_path}")
    raw = np.load(args.npy_path)
    print(f"原始形状: {raw.shape}, 范围: [{raw.min():.4f}, {raw.max():.4f}]")
    assert raw.shape[-1] == 263, f"Expected last dim=263, got {raw.shape[-1]}"

    # 263 -> joints22
    data_tensor = torch.from_numpy(raw).float().unsqueeze(0)  # (1, frames, 263)
    joints = recover_from_ric(data_tensor, joints_num=22)  # (1, frames, 22, 3)
    joints = joints.squeeze(0).numpy()  # (frames, 22, 3)
    print(f"转换后形状: {joints.shape}, 范围: [{joints.min():.4f}, {joints.max():.4f}]")

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.npy_path))[0]

    text_path = args.npy_path.replace('new_joint_vecs', 'texts').replace('.npy', '.txt')
    title = base_name
    if os.path.exists(text_path):
        with open(text_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                title = first_line.split('#')[0]
                print(f"动作描述: {title}")

    if args.mode in ['single', 'all']:
        fi = min(args.frame, joints.shape[0] - 1)
        plot_single_frame(joints[fi], T2M_KINEMATIC_CHAIN,
                          save_path=os.path.join(args.output_dir, f'{base_name}_frame{fi}.png'),
                          title=f"{title} (Frame {fi})")

    if args.mode in ['sequence', 'all']:
        plot_motion_sequence(joints, T2M_KINEMATIC_CHAIN,
                             save_path=os.path.join(args.output_dir, f'{base_name}_sequence.png'),
                             title=title)

    if args.mode in ['animation', 'all']:
        plot_motion_animation(joints, T2M_KINEMATIC_CHAIN,
                              save_path=os.path.join(args.output_dir, f'{base_name}_animation.gif'),
                              title=title, fps=args.fps)

    print("可视化完成!")


if __name__ == '__main__':
    main()
