import random
from omegaconf import OmegaConf
random.seed(42)
#lambda_reward = '1e0'

def float_to_1ex(x):
    if x == 0:
        return "0e0"
    s = f"{x:.0e}"  # 科学计数法格式化，如 "1e-01"
    return s.replace('+', '')

def ft_reset(ft_config):
    ft_type = ft_config['type']
    lambda_reward_1ex = float_to_1ex(ft_config['lambda_reward'])
    assert ft_type in ['ReFL', 'DRaFT', 'DRTune', 'AlignProp', 'NIPS'], f"Invalid type: {ft_type}"
    if ft_type == 'NIPS':
        k = ft_config['k']
        ft_config['enable_grad'] = [i for i in range(50)][-k:]
        return ft_config
    if ft_type == 'ReFL':
        ft_config['t_min'] = random.randint(50 - ft_config['M'] + 1, 50 - 1) # randint [a, b] is inclusive
        ft_config['enable_grad'] = [x for x in range(ft_config['t_min'], 50)]
        return ft_config
    elif ft_type == 'AlignProp':
        while True:
            enable_grad = [i for i in range(50) if random.random() < ft_config['prob']]
            if len(enable_grad) > 0:
                ft_config['enable_grad'] = enable_grad
                return ft_config
    elif ft_type == 'DRaFT':
        k = ft_config['k']
        if ft_config['custom'] is not None:
            ft_config['enable_grad'] = ft_config['custom']
            custom_str = '_['
            for i in ft_config['custom']:
                custom_str += str(i) + ','
            custom_str += + ']_'
            ft_config['name'] = f"DRaFT_C{custom_str}_R{lambda_reward_1ex}"
            
            return ft_config
        ft_config['name'] = f"DRaFT_K{k}_G{int(ft_config['skip'])}_Re{int(ft_config['reverse'])}_R{lambda_reward_1ex}"
        if ft_config['skip'] == True and ft_config['reverse'] == False:
            ft_config['enable_grad'] = [i for i in range(50)][::k]
            return ft_config
        elif ft_config['skip'] == False and ft_config['reverse'] == False:
            ft_config['enable_grad'] = [i for i in range(50)][-k:]
            ft_config['t_train'] = ft_config['enable_grad']
            return ft_config
        elif ft_config['skip'] == False and ft_config['reverse'] == True:
            ft_config['enable_grad'] = [i for i in range(50)][:k]
            return ft_config
        elif ft_config['skip'] == True and ft_config['reverse'] == True:
            ft_config['enable_grad'] = [i for i in range(50)][::-k]
            return ft_config
    elif ft_type == 'DRTune':
        ft_config['s'] = random.randint(0, ft_config['T'] - (ft_config['T'] // ft_config['k']) * ft_config['k'])
        ft_config['t_train'] = [ft_config['s'] + i * (ft_config['T'] // ft_config['k']) for i in range(ft_config['k'])]
        ft_config['t_min'] = random.randint(ft_config['T'] - ft_config['M'] + 1, 50 - 1) # randint [a, b] is inclusive
        ft_config['enable_grad'] = [i for i in range(50)]
        return ft_config
    
def get_ft_config(ft_type, m=1, prob=0.1, t=50, k=1, skip=False, reverse=False, custom=None, lambda_reward=1e0, dy=None, curriculum=False, sweep_ratio=0.03):
    """
    Get the finetune config based on the type of finetune.
    Args:
        ft_type (str): The type of finetune. Can be 'ReFL', 'DRaFT', 'DRTune', 'AlignProp'.
        m (int): The number of steps to finetune.
        prob (float): The probability of selecting a gradient.
        t (int): The number of steps to train.
        k (int): The number of steps to skip.
        skip (bool): Whether to skip the first k steps.
        reverse (bool): Whether to reverse the order of the gradients.
        custom (list): A list of custom gradients to select.
        lambda_reward (float): The Weight of lambda reward.
        curriculum (bool): Whether to use Motion Reward timestep scheduling (NIPS only).
    """
    assert ft_type in ['ReFL', 'DRaFT', 'DRTune', 'AlignProp', 'NIPS'], f"Invalid type: {ft_type}"
    if ft_type == 'ReFL':
        return ReFLClass(m, lambda_reward=lambda_reward)
    elif ft_type == 'AlignProp':
        return AlignPropClass(prob, lambda_reward=lambda_reward)
    elif ft_type == 'DRaFT':
        return DRaFTClass(k=k, reverse=reverse, skip=skip, custom=custom, lambda_reward=lambda_reward)
    elif ft_type == 'DRTune':
        return DRTuneClass(m=m, t=t, k=k, lambda_reward=lambda_reward)
    elif ft_type == 'NIPS':
        return NIPSClass(k=k, lambda_reward=lambda_reward, dy=dy, curriculum=curriculum, sweep_ratio=sweep_ratio)
    
def NIPSClass(k, lambda_reward, dy=None, curriculum=False, sweep_ratio=0.03):
    NIPS = {}
    NIPS['type'] = 'NIPS'
    NIPS['k'] = k
    lambda_reward_1ex = float_to_1ex(lambda_reward)
    NIPS['curriculum'] = curriculum
    NIPS['sweep_ratio'] = sweep_ratio
    if curriculum:
        NIPS['name'] = f"MR_K{k}_SW{sweep_ratio}_R{lambda_reward_1ex}"
    else:
        NIPS['name'] = f"NIPS_K{k}_R{lambda_reward_1ex}"
    NIPS['lambda_reward'] = float(lambda_reward)
    NIPS['dy'] = dy
    NIPS = ft_reset(NIPS)
    return NIPS


def motion_reward_curriculum_update(ft_config, progress, total_steps=50):
    """
    Motion Reward 时间步调度 (非均匀分段线性)
    
    根据训练进度 progress ∈ [0, 1]，动态滑动 enable_grad 窗口：
      - 早期优化高噪声范围 (step 索引小, i≈0)
      - 后期优化低噪声范围 (step 索引大, i≈49)
    
    sweep_ratio 从 ft_config['sweep_ratio'] 读取：
      - sweep_ratio=0.0 → 100% 时间固定在最后 k 步 (纯低噪声)
      - sweep_ratio=0.03 → 3% 扫过全部步骤 + 97% 固定最后 k 步
      - sweep_ratio=0.1  → 10% 扫过 + 90% 固定最后 k 步
    
    例 (total_steps=50, k=10, sweep_ratio=0.03):
      progress 0~0.03: 窗口 [0,9] → ... → [40,49]  (密集快扫)
      progress 0.03~1.0: 窗口固定 [40,49]             (集中精炼)
    
    Args:
        ft_config: NIPS ft_config dict (会被原地修改，需含 'k' 和 'sweep_ratio')
        progress: float ∈ [0, 1]，训练进度比例
        total_steps: diffusion 总步数，默认50
    Returns:
        ft_config (原地修改后返回)
    """
    k = ft_config['k']
    sweep_ratio = ft_config.get('sweep_ratio', 0.03)
    progress = max(0.0, min(1.0, progress))
    max_start = total_steps - k
    
    if sweep_ratio <= 0.0 or progress > sweep_ratio:
        start = max_start
    else:
        start = int(round((progress / sweep_ratio) * max_start))
    
    start = max(0, min(start, max_start))
    ft_config['enable_grad'] = list(range(start, start + k))
    return ft_config



def AlignPropClass(prob, lambda_reward):
    AlignProp = {}
    AlignProp['type'] = 'AlignProp'
    AlignProp['prob'] = prob
    lambda_reward_1ex = float_to_1ex(lambda_reward)
    AlignProp['name'] = f"AlignProp_P{int(prob*100)}_R{lambda_reward_1ex}"
    if prob < 1e-3:
        assert 1 == 2, 'prob < 1e-3 is not supported'
    AlignProp['lambda_reward'] = float(lambda_reward)
    AlignProp = ft_reset(AlignProp)
    return AlignProp
        

def DRaFTClass(k, reverse=False, skip=False, custom=None, lambda_reward=None, maxT=50):
    DRaFT = {}
    DRaFT['type'] = 'DRaFT'
    DRaFT['k'] = k
    DRaFT['maxT'] = maxT
    DRaFT['skip'] = skip
    DRaFT['reverse'] = reverse
    DRaFT['custom'] = custom
    DRaFT['lambda_reward'] = float(lambda_reward)
    DRaFT = ft_reset(DRaFT)  
    return DRaFT
        
def DRTuneClass(m, t, k, lambda_reward, maxT=50):
    DRTune = {}
    DRTune['type'] = 'DRTune'
    DRTune['M'] = m
    DRTune['t_min'] = random.randint(maxT - DRTune['M'] + 1, 50 - 1) # randint [a, b] is inclusive
    DRTune['T'] = t
    DRTune['k'] = k
    DRTune['lambda_reward'] = float(lambda_reward)
    lambda_reward_1ex = float_to_1ex(lambda_reward)
    DRTune['name'] = f"DRTune_T{DRTune['T']}_K{DRTune['k']}_M{DRTune['t_min']}_R{lambda_reward_1ex}"
    DRTune = ft_reset(DRTune)
    return DRTune
    
def ReFLClass(m, lambda_reward):
    ReFL = {}
    ReFL['type'] = 'ReFL'
    ReFL['M'] = m
    lambda_reward_1ex = float_to_1ex(lambda_reward)
    ReFL['name'] = f"ReFL_M{ReFL['M']}_R{lambda_reward_1ex}"
    ReFL['lambda_reward'] = float(lambda_reward)
    ReFL = ft_reset(ReFL)
    return ReFL
    
        
        


