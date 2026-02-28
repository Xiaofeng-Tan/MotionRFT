"""Test dtype after creating CriticRewardAdapter"""
import sys, os, torch
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from reward_adapter import process_T5_outputs
from motionrft_mld import CriticRewardAdapter

device = torch.device('cuda')

sys.argv = ['test', '--cfg', 'configs/ft_mld_t2m.yaml']
cfg = parse_args()

dataset = get_dataset(cfg)
model = MLD(cfg, dataset)

state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict, strict=False)
model.to(device)

print("=== Before creating CriticRewardAdapter ===")
for name, param in model.text_encoder.text_model.named_parameters():
    print(f'  T5 {name}: {param.dtype}')
    break

# Create CriticRewardAdapter (this creates another T5 model)
print("\n=== Creating CriticRewardAdapter ===")
reward_model = CriticRewardAdapter(
    backbone_ckpt='../checkpoints/motionreward/stage1_retrieval_backbone_r128.pth',
    critic_lora_ckpt='../checkpoints/motionreward/stage2_critic_lora_r128.pth',
    critic_head_ckpt='../checkpoints/motionreward/stage2_critic_head_r128.pth',
    ai_detection_lora_ckpt='../checkpoints/motionreward/stage3_ai_detection_lora_r128.pth',
    ai_detection_head_ckpt='../checkpoints/motionreward/stage3_ai_detection_head_r128.pth',
    t5_path='../deps/sentence-t5-large',
    model_size='tiny',
    lora_rank=128,
    lora_alpha=256,
    device='cuda',
    lambda_critic=0.0002,
    lambda_retrieval=1.0,
    lambda_m2m=1.0,
    lambda_ai_detection=0.0002,
)

print("\n=== After creating CriticRewardAdapter ===")
for name, param in model.text_encoder.text_model.named_parameters():
    print(f'  MLD T5 {name}: {param.dtype}')
    break

# Test text embedding
texts = ['a person walks forward', '']
t_len, token_emb, text_emb = process_T5_outputs(texts, model.text_encoder.text_model)
print(f'  text_emb dtype: {text_emb.dtype}')

encoder_hidden_states = text_emb.permute(1, 0, 2)
print(f'  encoder_hidden_states dtype: {encoder_hidden_states.dtype}')
try:
    result = model.denoiser.emb_proj(encoder_hidden_states)
    print(f'  emb_proj result dtype: {result.dtype}')
    print('No error!')
except RuntimeError as e:
    print(f'  ERROR: {e}')
