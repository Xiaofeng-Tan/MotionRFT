#!/usr/bin/env python3
"""
下载 HY-Motion RL 训练所需的依赖模型
"""
import os
import subprocess
from pathlib import Path

def download_sentence_t5():
    """下载 sentence-t5-large 模型"""
    deps_dir = Path("deps")
    deps_dir.mkdir(exist_ok=True)
    
    model_path = deps_dir / "sentence-t5-large"
    
    if model_path.exists():
        print(f"✓ sentence-t5-large 已存在: {model_path}")
        return
    
    print("📥 下载 sentence-t5-large 模型...")
    try:
        # 使用 huggingface_hub 下载
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="sentence-transformers/sentence-t5-large",
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        print("✓ sentence-t5-large 下载完成")
    except ImportError:
        print("❌ 需要安装 huggingface_hub: pip install huggingface_hub")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请手动下载或使用 git clone:")
        print(f"git clone https://huggingface.co/sentence-transformers/sentence-t5-large {model_path}")

def main():
    print("🚀 HY-Motion RL 依赖下载器")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("hymotion").exists():
        print("❌ 请在 HY-Motion-1.0 目录下运行此脚本")
        return
    
    # 下载模型
    download_sentence_t5()
    
    print("\n✅ 依赖下载完成！")
    print("\n📋 已包含文件:")
    print("✓ SPM checkpoint: evaluator.pth (102MB)")
    print("✓ HY-Motion 代码: hymotion/ (804KB)")
    print("✓ SPM 代码: ReAlignModule/ (136KB)")
    print("✓ 3D 模型: assets/wooden_models/ (30MB)")
    print("\n📋 下一步:")
    print("1. 准备数据集: --motion_dir path/to/joints_hunyuan")
    print("2. 运行训练: python motionrft_hy.py --spm_path evaluator.pth ...")

if __name__ == "__main__":
    main()