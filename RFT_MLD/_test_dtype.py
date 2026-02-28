"""Quick test to reproduce the dtype mismatch"""
import sys, os, torch
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from reward_adapter import process_T5_outputs

# Minimal test
device = torch.device('cuda')

# Parse config
sys.argv = ['test', '--cfg', 'configs/ft_mld_t2m.yaml']
cfg = parse_args()

dataset = get_dataset(cfg)
model = MLD(cfg, dataset)

# Load pretrained
state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict, strict=False)
model.to(device)

# Check text encoder dtype
for name, param in model.text_encoder.text_model.named_parameters():
    print(f'text_encoder T5 {name}: {param.dtype} {param.device}')
    break

# Check denoiser emb_proj dtype
for name, param in model.denoiser.named_parameters():
    if 'emb_proj' in name:
        print(f'denoiser {name}: {param.dtype} {param.device}')
        break

# Process text
texts = ['a person walks forward', '']
t_len, token_emb, text_emb = process_T5_outputs(texts, model.text_encoder.text_model)
print(f'text_emb dtype: {text_emb.dtype}, device: {text_emb.device}')

# Test emb_proj
encoder_hidden_states = text_emb.permute(1, 0, 2)
print(f'encoder_hidden_states dtype: {encoder_hidden_states.dtype}')
result = model.denoiser.emb_proj(encoder_hidden_states)
print(f'emb_proj result dtype: {result.dtype}')
print('No error!')
