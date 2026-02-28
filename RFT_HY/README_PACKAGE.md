# HY-Motion RL 训练包

## 包含内容

| 文件/目录 | 大小 | 说明 |
|-----------|------|------|
| `evaluator.pth` | 102MB | **SPM Reward Model** 预训练权重 |
| `hymotion/` | 804KB | HY-Motion 核心代码 |
| `ReAlignModule/` | 136KB | SPM 模型定义代码 |
| `assets/wooden_models/` | 30MB | 3D 可视化模型 |
| `*.py` | <1MB | 训练脚本和工具 |
| `download_dependencies.py` | - | 自动下载 sentence-t5-large |

**总大小：~124MB**（不含 sentence-t5-large）

## 快速开始

### 1. 解压并设置
```bash
unzip HY-Motion-RL-without-sentence-t5.zip
cd HY-Motion-1.0/
```

### 2. 下载 sentence-t5-large 文本编码器
```bash
python download_dependencies.py
```
或手动下载：
```bash
mkdir -p deps
git clone https://huggingface.co/sentence-transformers/sentence-t5-large deps/sentence-t5-large
```

### 3. 准备数据集
确保有 HumanML3D 数据集的 `joints_hunyuan` 格式。

### 4. 运行训练
```bash
python motionrft_hy.py \
    --model_path ckpts/tencent/HY-Motion-1.0-Lite/ \
    --spm_path evaluator.pth \
    --motion_dir /path/to/joints_hunyuan \
    --device_ids 0 \
    --batch_size 4 \
    --num_epochs 100 \
    --train_split train
```

## 依赖要求

```bash
pip install torch>=2.1.0 transformers>=4.38.0 sentence-transformers==2.2.2
pip install einops safetensors tqdm pyyaml omegaconf swanlab
```

## 文件说明

- `motionrft_hy.py`: 主训练脚本
- `evaluator.pth`: SPM reward model，用于计算 text-motion 和 motion-motion 相似度
- `ReAlignModule/`: SPM 模型的实现代码
- `hymotion/`: HY-Motion 扩散模型的实现
- `assets/wooden_models/`: 3D 人体模型，用于可视化

## 注意事项

1. **SPM checkpoint 路径**: 已包含在压缩包根目录，使用 `--spm_path evaluator.pth`
2. **sentence-t5-large**: 需要单独下载（约6.7GB），运行 `download_dependencies.py` 自动下载
3. **HunyuanMotion 模型**: 需要单独下载到 `ckpts/tencent/HY-Motion-1.0-Lite/`
4. **数据集**: 需要 HumanML3D 的 `joints_hunyuan` 格式数据