#!/usr/bin/env python3
"""
Visualize packed .pth data (Critic & AI Detection) by sampling motions
and rendering stick-figure animations.

Usage:
    python scripts/_vis_packed.py                          # default: all packed files
    python scripts/_vis_packed.py --data_root /path/to/datasets
    python scripts/_vis_packed.py --num_samples 5
"""

import os
import sys
import argparse
import torch
import numpy as np
import subprocess
import random
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_263 = os.path.join(SCRIPT_DIR, 'visualize_263.py')
VIS_22x3 = os.path.join(SCRIPT_DIR, 'visualize_joints22.py')
VIS_135 = os.path.join(SCRIPT_DIR, 'visualize_135.py')

VIS_MAP = {'263': VIS_263, '22x3': VIS_22x3, '135': VIS_135}


def infer_repr_from_filename(filename):
    """Infer representation type from filename, e.g. critic_val_22x3.pth -> 22x3"""
    base = os.path.splitext(os.path.basename(filename))[0]
    for repr_type in ['22x3', '263', '135']:
        if repr_type in base:
            return repr_type
    return None


def infer_repr_from_shape(motion):
    """Infer representation type from motion tensor shape."""
    if isinstance(motion, torch.Tensor):
        shape = motion.shape
    else:
        shape = motion.shape
    # Remove leading batch dims of size 1
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    last_dim = shape[-1]
    if last_dim == 263:
        return '263'
    elif last_dim == 135:
        return '135'
    elif last_dim == 3 and len(shape) >= 2:
        return '22x3'
    return None


def squeeze_motion(motion):
    """Remove leading batch dims of size 1 from motion."""
    if isinstance(motion, torch.Tensor):
        while motion.dim() > 2 and motion.shape[0] == 1:
            motion = motion.squeeze(0)
        return motion.numpy()
    else:
        while motion.ndim > 2 and motion.shape[0] == 1:
            motion = motion.squeeze(0)
        return motion


def visualize_ai_detection(pth_path, output_base, tmp_dir, num_samples=3):
    """Visualize AI Detection packed data: {repr_type: {motions, lengths, sources}}"""
    print(f"\n{'='*60}")
    print(f"AI Detection: {os.path.basename(pth_path)}")
    print(f"{'='*60}")
    data = torch.load(pth_path, map_location='cpu', weights_only=False)
    print(f"Repr types: {list(data.keys())}")

    for repr_type in data:
        motions = data[repr_type]['motions']
        lengths = data[repr_type]['lengths']
        sources = data[repr_type]['sources']
        n = motions.shape[0]
        print(f"\n  [{repr_type}] {n} samples, shape: {motions.shape}")

        out_dir = os.path.join(output_base, f'ai_detection_{repr_type}')
        os.makedirs(out_dir, exist_ok=True)

        random.seed(42)
        indices = random.sample(range(n), min(num_samples, n))
        for idx in indices:
            length = lengths[idx].item()
            motion = squeeze_motion(motions[idx][:length])
            tag = f"ai_det_{repr_type}_idx{idx}_{sources[idx]}"
            _run_vis(motion, repr_type, tag, out_dir, tmp_dir)


def visualize_critic(pth_path, output_base, tmp_dir, num_samples=3):
    """Visualize Critic paired data: list of {motion_better, motion_worse, ...}"""
    # Infer repr_type from filename (e.g. critic_val_22x3.pth -> 22x3)
    repr_from_file = infer_repr_from_filename(pth_path)

    print(f"\n{'='*60}")
    print(f"Critic: {os.path.basename(pth_path)} (repr={repr_from_file})")
    print(f"{'='*60}")
    data = torch.load(pth_path, map_location='cpu', weights_only=False)

    out_dir = os.path.join(output_base, f'critic_{repr_from_file or "unknown"}')
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(data, list):
        print(f"  {len(data)} pairs")
        random.seed(42)
        indices = random.sample(range(len(data)), min(num_samples, len(data)))
        for idx in indices:
            pair = data[idx]
            # Use filename-based repr first, then data field, then shape inference
            repr_type = repr_from_file or pair.get('repr_type', None)
            for key in ['motion_better', 'motion_worse']:
                if key not in pair:
                    continue
                motion = pair[key]
                if isinstance(motion, torch.Tensor):
                    motion = motion.clone()
                else:
                    motion = np.array(motion)
                length_key = f'length_{key.split("_")[1]}'
                length = pair.get(length_key, None)
                if length is not None:
                    if isinstance(length, torch.Tensor):
                        length = length.item()
                    motion = motion[:int(length)]
                motion = squeeze_motion(motion)
                # Final repr inference from shape if still unknown
                if repr_type is None:
                    repr_type = infer_repr_from_shape(motion)
                tag = f"critic_{repr_type}_{key}_idx{idx}"
                _run_vis(motion, repr_type, tag, out_dir, tmp_dir)
    elif isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        for key in data:
            rdata = data[key]
            repr_type = repr_from_file or key
            if isinstance(rdata, dict) and 'motions' in rdata:
                motions = rdata['motions']
                lengths = rdata.get('lengths', None)
                n = motions.shape[0]
                print(f"  [{repr_type}] {n} samples, shape: {motions.shape}")
                random.seed(42)
                indices = random.sample(range(n), min(num_samples, n))
                for idx in indices:
                    motion = motions[idx]
                    if lengths is not None:
                        motion = motion[:lengths[idx].item()]
                    motion = squeeze_motion(motion)
                    tag = f"critic_{repr_type}_idx{idx}"
                    _run_vis(motion, repr_type, tag, out_dir, tmp_dir)


def _run_vis(motion, repr_type, tag, output_dir, tmp_dir):
    """Save motion as .npy and call the appropriate vis script."""
    vis_script = VIS_MAP.get(repr_type)
    if vis_script is None or not os.path.isfile(vis_script):
        print(f"    [Skip] No vis script for repr_type={repr_type}")
        return

    npy_path = os.path.join(tmp_dir, f"{tag}.npy")
    np.save(npy_path, motion)
    print(f"    {tag}: shape={motion.shape}")

    cmd = [sys.executable, vis_script,
           '--npy_path', npy_path,
           '--output_dir', output_dir,
           '--mode', 'animation',
           '--fps', '20']
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Visualize Critic & AI Detection packed data')
    parser.add_argument('--data_root', type=str,
                        default=os.path.join(SCRIPT_DIR, '..', 'datasets'),
                        help='Root directory containing critic/ and ai_detection_packed/')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output base directory (default: <data_root>/visualizations/)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize per repr type')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_base = args.output_dir or os.path.join(data_root, 'visualizations')
    tmp_dir = os.path.join(output_base, '_tmp_npy')
    os.makedirs(tmp_dir, exist_ok=True)

    # Critic data (one file per repr type)
    critic_dir = os.path.join(data_root, 'critic')
    if os.path.isdir(critic_dir):
        for fname in ['critic_val_263.pth', 'critic_val_22x3.pth', 'critic_val_135.pth']:
            fpath = os.path.join(critic_dir, fname)
            if os.path.isfile(fpath):
                visualize_critic(fpath, output_base, tmp_dir, args.num_samples)

    # AI Detection data
    ai_dir = os.path.join(data_root, 'ai_detection_packed')
    if os.path.isdir(ai_dir):
        for fname in ['ai_generated_val.pth']:
            fpath = os.path.join(ai_dir, fname)
            if os.path.isfile(fpath):
                visualize_ai_detection(fpath, output_base, tmp_dir, args.num_samples)

    # Cleanup temp files
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    print(f"\nAll visualizations saved to: {output_base}")


if __name__ == '__main__':
    main()
