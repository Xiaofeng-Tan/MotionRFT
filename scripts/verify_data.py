#!/usr/bin/env python3
"""
Verify MotionRFT dataset integrity and visualize samples.

Usage:
    # Verify all datasets
    python scripts/verify_data.py --data_root ./datasets

    # Verify + visualize samples
    python scripts/verify_data.py --data_root ./datasets --visualize --output_dir ./visualizations
"""

import argparse
import os
import sys
import numpy as np
import torch


def check_humanml3d(data_root):
    """Verify HumanML3D multi-representation data."""
    base = os.path.join(data_root, "humanml3d")
    print("\n" + "=" * 60)
    print("Checking HumanML3D data")
    print("=" * 60)

    checks = {
        "263-dim (new_joint_vecs)": ("new_joint_vecs", 263),
        "22x3 joints (new_joints)": ("new_joints", None),  # shape varies
        "135-dim 6D rotation (joints_6d)": ("joints_6d", 135),
        "Text annotations (texts)": ("texts", None),
    }

    stats = {}
    for name, (subdir, expected_dim) in checks.items():
        path = os.path.join(base, subdir)
        if not os.path.isdir(path):
            print(f"  [FAIL] {name}: directory not found at {path}")
            continue

        files = os.listdir(path)
        n_files = len(files)
        print(f"  [OK] {name}: {n_files} files")

        # Sample check for npy files
        if expected_dim is not None:
            npy_files = [f for f in files if f.endswith(".npy")]
            if npy_files:
                sample = np.load(os.path.join(path, npy_files[0]))
                print(f"       Sample shape: {sample.shape}, dtype: {sample.dtype}")
                if sample.shape[-1] != expected_dim:
                    print(f"       [WARN] Expected dim {expected_dim}, got {sample.shape[-1]}")
                stats[subdir] = {"n_files": len(npy_files), "sample_shape": sample.shape}
        elif subdir == "new_joints":
            npy_files = [f for f in files if f.endswith(".npy")]
            if npy_files:
                sample = np.load(os.path.join(path, npy_files[0]))
                print(f"       Sample shape: {sample.shape}, dtype: {sample.dtype}")
                stats[subdir] = {"n_files": len(npy_files), "sample_shape": sample.shape}

    # Check split files
    for split_file in ["train.txt", "val.txt", "test.txt", "all.txt"]:
        fpath = os.path.join(base, split_file)
        if os.path.isfile(fpath):
            with open(fpath) as f:
                lines = f.readlines()
            print(f"  [OK] {split_file}: {len(lines)} entries")
        else:
            print(f"  [FAIL] {split_file}: not found")

    # Check mean/std files
    for stat_file in ["Mean.npy", "Std.npy", "Mean_22x3.npy", "Std_22x3.npy", "Mean_135.npy", "Std_135.npy"]:
        fpath = os.path.join(base, stat_file)
        if os.path.isfile(fpath):
            arr = np.load(fpath)
            print(f"  [OK] {stat_file}: shape={arr.shape}")
        else:
            print(f"  [FAIL] {stat_file}: not found")

    return stats


def check_critic(data_root):
    """Verify Critic (preference) data."""
    base = os.path.join(data_root, "critic")
    print("\n" + "=" * 60)
    print("Checking Critic data")
    print("=" * 60)

    files = [
        "critic_train_263.pth", "critic_train_22x3.pth", "critic_train_135.pth",
        "critic_val_263.pth", "critic_val_22x3.pth", "critic_val_135.pth",
    ]

    for fname in files:
        fpath = os.path.join(base, fname)
        if os.path.isfile(fpath):
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                keys = list(data.keys())
                print(f"  [OK] {fname}: keys={keys}")
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        print(f"       {k}: shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(v, list):
                        print(f"       {k}: list, len={len(v)}")
            elif isinstance(data, list):
                print(f"  [OK] {fname}: list with {len(data)} entries")
                if len(data) > 0:
                    sample = data[0]
                    print(f"       Sample type: {type(sample)}")
            else:
                print(f"  [OK] {fname}: type={type(data)}")
        else:
            print(f"  [FAIL] {fname}: not found")


def check_ai_detection(data_root):
    """Verify AI detection data."""
    base = os.path.join(data_root, "ai_detection_packed")
    print("\n" + "=" * 60)
    print("Checking AI Detection data")
    print("=" * 60)

    files = ["ai_generated_train.pth", "ai_generated_val.pth", "ai_generated_test.pth"]

    for fname in files:
        fpath = os.path.join(base, fname)
        if os.path.isfile(fpath):
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                repr_types = list(data.keys())
                print(f"  [OK] {fname}: repr_types={repr_types}")
                for rtype in repr_types:
                    rdata = data[rtype]
                    if isinstance(rdata, dict):
                        for k, v in rdata.items():
                            if isinstance(v, torch.Tensor):
                                print(f"       [{rtype}] {k}: shape={v.shape}")
                            elif isinstance(v, list):
                                print(f"       [{rtype}] {k}: list, len={len(v)}")
            else:
                print(f"  [OK] {fname}: type={type(data)}")
        else:
            print(f"  [FAIL] {fname}: not found")


def visualize_samples(data_root, output_dir):
    """Visualize sample motions from each representation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    os.makedirs(output_dir, exist_ok=True)

    T2M_KINEMATIC_CHAIN = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]
    colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]

    def plot_skeleton(joints, title, save_path):
        """Plot a single skeleton frame."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="black", s=30)
        for chain, color in zip(T2M_KINEMATIC_CHAIN, colors):
            ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2],
                      linewidth=3, color=color)
        ax.set_title(title)
        max_range = np.array([joints[:, d].max() - joints[:, d].min() for d in range(3)]).max() / 2.0
        mid = [(joints[:, d].max() + joints[:, d].min()) * 0.5 for d in range(3)]
        for d, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            setter(mid[d] - max_range, mid[d] + max_range)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # 1. Visualize 22x3 joints
    joints_dir = os.path.join(data_root, "humanml3d", "new_joints")
    if os.path.isdir(joints_dir):
        npy_files = sorted([f for f in os.listdir(joints_dir) if f.endswith(".npy")])[:3]
        for f in npy_files:
            data = np.load(os.path.join(joints_dir, f))
            mid_frame = data.shape[0] // 2
            frame = data[mid_frame]
            name = os.path.splitext(f)[0]
            plot_skeleton(frame, f"22x3 joints: {name} (frame {mid_frame})",
                         os.path.join(output_dir, f"vis_22x3_{name}.png"))

    # 2. Visualize 263-dim (convert to joints first)
    vecs_dir = os.path.join(data_root, "humanml3d", "new_joint_vecs")
    if os.path.isdir(vecs_dir):
        npy_files = sorted([f for f in os.listdir(vecs_dir) if f.endswith(".npy")])[:3]
        for f in npy_files:
            data = np.load(os.path.join(vecs_dir, f))
            tensor = torch.from_numpy(data).float().unsqueeze(0)
            # recover_from_ric
            from _recover_joints import recover_from_ric
            joints = recover_from_ric(tensor, joints_num=22).squeeze(0).numpy()
            mid_frame = joints.shape[0] // 2
            name = os.path.splitext(f)[0]
            plot_skeleton(joints[mid_frame], f"263-dim: {name} (frame {mid_frame})",
                         os.path.join(output_dir, f"vis_263_{name}.png"))

    print(f"\nVisualization complete! Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Verify MotionRFT dataset integrity")
    parser.add_argument("--data_root", type=str, default="./datasets",
                        help="Root directory of datasets")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization of sample motions")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="Output directory for visualizations")
    args = parser.parse_args()

    print("MotionRFT Dataset Verification")
    print("=" * 60)
    print(f"Data root: {os.path.abspath(args.data_root)}")

    check_humanml3d(args.data_root)
    check_critic(args.data_root)
    check_ai_detection(args.data_root)

    if args.visualize:
        visualize_samples(args.data_root, args.output_dir)

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
