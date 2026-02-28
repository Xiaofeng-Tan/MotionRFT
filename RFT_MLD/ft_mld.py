import os
import inspect
import sys
import logging
import datetime
import os.path as osp
import glob
import random

from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import swanlab
import diffusers
import transformers
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import print_table, set_seed, move_batch_to_device
from mld.data.humanml.utils.plot_script import plot_3d_motion

# 使用新的 reward adapter 替代原来的 SPM
from reward_adapter import MotionRewardAdapter, process_T5_outputs
from ft_config import get_ft_config

# 添加 reward model 评估所需的导入
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motionreward.evaluation.retrieval_metrics import calculate_retrieval_metrics, calculate_retrieval_metrics_small_batches

import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from torch.profiler import profile, record_function, ProfilerActivity


def fuck(model):
    para_cnt, para_sum = {}, {}
    for name, para in model.named_parameters():
        try:
            para_cnt[name.split('.')[0]] += para.sum().item()
            para_sum[name.split('.')[0]] += para.numel()
        except:
            para_cnt[name.split('.')[0]] = para.sum().item()
            para_sum[name.split('.')[0]] = para.numel()
    print(para_cnt, '\n', para_sum)

    return para_cnt, para_sum
def main():
    
    cfg = parse_args()
    
    ft_config = get_ft_config(ft_type=cfg.ft_type, m=cfg.ft_m, prob=cfg.ft_prob, t=cfg.ft_t, k=cfg.ft_k, \
                              skip=cfg.ft_skip, reverse=cfg.ft_reverse, custom=None, lambda_reward=cfg.ft_lambda_reward, dy=cfg.ft_dy)
    
    # 设置 reward model checkpoint 路径
    # 支持绝对路径和相对路径
    if cfg.spm_path and os.path.isabs(cfg.spm_path):
        ft_config['reward_model_ckpt'] = cfg.spm_path
    elif cfg.spm_path:
        # 优先查找 motionreward，然后查找 spm_lora_models/retrieval_lora_models
        motionreward_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'checkpoints', 'motionreward', f'{cfg.spm_path}.pth'
        )
        spm_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            '..', 'checkpoints', 'spm_lora_models', f'{cfg.spm_path}.pth'
        )
        retrieval_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            '..', 'checkpoints', 'retrieval_lora_models', f'{cfg.spm_path}.pth'
        )
        if os.path.exists(motionreward_path):
            ft_config['reward_model_ckpt'] = motionreward_path
        elif os.path.exists(spm_path):
            ft_config['reward_model_ckpt'] = spm_path
        elif os.path.exists(retrieval_path):
            ft_config['reward_model_ckpt'] = retrieval_path
        else:
            ft_config['reward_model_ckpt'] = cfg.spm_path  # 直接使用原路径
    else:
        # 默认使用 motionreward checkpoint
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'checkpoints', 'motionreward'
        )
        stage2_files = glob.glob(os.path.join(checkpoint_dir, '*_stage2_best_retrieval_backbone_*.pth'))
        if stage2_files:
            ft_config['reward_model_ckpt'] = max(stage2_files, key=os.path.getmtime)
            print(f"[Auto-selected] Using Stage 2 checkpoint: {ft_config['reward_model_ckpt']}")
        else:
            stage1_files = glob.glob(os.path.join(checkpoint_dir, '*_stage1_best_retrieval_backbone_*.pth'))
            if stage1_files:
                ft_config['reward_model_ckpt'] = max(stage1_files, key=os.path.getmtime)
                print(f"[Auto-selected] Using Stage 1 checkpoint: {ft_config['reward_model_ckpt']}")
            else:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    # ft_config['name'] += f'_{cfg.spm_path}_{cfg.ft_lr}'

    ft_config['name'] += f'_lr{cfg.ft_lr:.0e}'
    if cfg.ft_dy != 2:
        ft_config['name'] += f'_dy{cfg.ft_dy}'
    
    # 设置 reward model size
    ft_config['reward_model_size'] = getattr(cfg, 'reward_model_size', 'base')
    
    cfg.ft_config = ft_config

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.SEED_VALUE)

    #name_time_str = osp.join(cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    #time_str = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    #cfg.output_dir = osp.join(cfg.FOLDER, name_time_str)
    mld_path, spm_path = cfg.mld_path, cfg.spm_path
    #cfg.output_dir = f'./checkpoints/ft_mld/{spm_path}_{time_str}'
    # cfg.output_dir = f"./checkpoints/ft_mld_step_iclr_re/{spm_path}/{ft_config['name']}"
    cfg.output_dir = f"./checkpoints/ft_mld_step_iclr_re_final/{ft_config['name']}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=True)


    if cfg.vis == "tb":
        writer = SummaryWriter(cfg.output_dir)
    elif cfg.vis == "swanlab":
        # 将 OmegaConf 转换为可序列化的字典
        # 由于配置中存在无法解析的插值引用（如 ${DATASET.NFEATS}），
        # 我们只记录关键配置信息
        cfg_dict = {
            'output_dir': cfg.output_dir,
            'ft_type': cfg.ft_type,
            'ft_m': cfg.ft_m,
            'ft_lr': cfg.ft_lr,
            'ft_lambda_reward': cfg.ft_lambda_reward,
            'spm_path': str(cfg.get('spm_path', '')),
            'TRAIN': {
                'BATCH_SIZE': cfg.TRAIN.BATCH_SIZE,
                'learning_rate': cfg.TRAIN.learning_rate,
                'max_ft_epochs': cfg.TRAIN.max_ft_epochs,
            },
            'VAL': {
                'BATCH_SIZE': cfg.VAL.BATCH_SIZE,
                'SPLIT': cfg.VAL.SPLIT,
            },
            'DATASET': {
                'NAME': cfg.DATASET.NAME,
            }
        }
        writer = swanlab.init(project="MotionLCM",
                              experiment_name=os.path.normpath(cfg.output_dir).replace(os.path.sep, "-"),
                              suffix=None, config=cfg_dict, logdir=cfg.output_dir)
    else:
        raise ValueError(f"Invalid vis method: {cfg.vis}")

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(cfg.output_dir, 'output.log'))
    handlers = [file_handler, stream_handler]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    
    OmegaConf.save(cfg, osp.join(cfg.output_dir, 'config.yaml'))
    

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    dataset = get_dataset(cfg)
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()

    model = MLD(cfg, dataset)

    assert cfg.TRAIN.PRETRAINED, "cfg.TRAIN.PRETRAINED must not be None."
    logger.info(f"Loading pre-trained model: {cfg.TRAIN.PRETRAINED}")
    state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
    logger.info(model.load_state_dict(state_dict, strict=False))

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.vae.eval()
    model.text_encoder.eval()
    model.to(device)

    cfg.TRAIN.learning_rate = float(cfg.ft_lr)
    logger.info("learning_rate: {}".format(cfg.TRAIN.learning_rate))
    optimizer = torch.optim.AdamW(
        model.denoiser.parameters(),
        lr=cfg.TRAIN.learning_rate,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon)
    if cfg.TRAIN.max_ft_steps == -1:
        cfg.TRAIN.max_ft_steps = cfg.TRAIN.max_ft_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=cfg.TRAIN.max_ft_steps)

    # Train!
    logger.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logging.info(f"  Num Epochs = {cfg.TRAIN.max_ft_epochs}")
    logging.info(f"  Instantaneous batch size per device = {cfg.TRAIN.BATCH_SIZE}")
    logging.info(f"  Total optimization steps = {cfg.TRAIN.max_ft_steps}")
    logging.info(f"  1 Epoch == {len(train_dataloader)} Step")

    global_step = 0

    @torch.no_grad()
    def validation(target_model: MLD, ema: bool = False) -> tuple:
        target_model.denoiser.eval()
        val_loss_list = []
        
        # 随机选择一个 batch 用于可视化
        vis_batch_idx = random.randint(0, len(val_dataloader) - 1)
        vis_joints = None
        vis_text = None
        
        for batch_idx, val_batch in enumerate(tqdm(val_dataloader)):
            val_batch = move_batch_to_device(val_batch, device)
            val_loss_dict = target_model.allsplit_step(split='val', batch=val_batch)
            val_loss_list.append(val_loss_dict)
            
            # 在选中的 batch 生成可视化（使用 t2m_eval 的结果）
            if batch_idx == vis_batch_idx and vis_joints is None:
                try:
                    # 调用 t2m_eval 获取生成的关节坐标
                    rs_set = target_model.t2m_eval(val_batch)
                    joints_rst = rs_set['joints_rst']  # [B, T, 22, 3]
                    
                    # 随机选择 batch 中的一个样本
                    batch_size = joints_rst.shape[0]
                    sample_idx = random.randint(0, batch_size - 1)
                    
                    # 获取文本
                    text = val_batch['text'][sample_idx]
                    if isinstance(text, list):
                        text = text[0]
                    length = val_batch['length'][sample_idx]
                    if isinstance(length, torch.Tensor):
                        length = length.item()
                    
                    # 获取关节坐标并截断到实际长度
                    vis_joints = joints_rst[sample_idx, :length].cpu().numpy()  # [T, 22, 3]
                    vis_text = text
                    logger.info(f"Generated motion for visualization: '{text[:50]}...' with shape {vis_joints.shape}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate visualization: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 保存可视化视频
        if vis_joints is not None:
            vis_dir = os.path.join(cfg.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 生成文件名（清理特殊字符）
            safe_text = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in vis_text[:30])
            safe_text = safe_text.replace(' ', '_')
            video_path = os.path.join(vis_dir, f"step{global_step}_{safe_text}.mp4")
            
            try:
                plot_3d_motion(
                    save_path=video_path,
                    joints=vis_joints,
                    title=f"Step {global_step}: {vis_text[:50]}",
                    fps=20
                )
                logger.info(f"Saved visualization video: {video_path}")
                
                # 记录到 swanlab/tensorboard
                if cfg.vis == "swanlab":
                    writer.log({"visualization/motion_video": swanlab.Video(video_path)}, step=global_step)
                elif cfg.vis == "tb":
                    writer.add_text("visualization/video_path", video_path, global_step=global_step)
                    
            except Exception as e:
                logger.warning(f"Failed to save visualization: {e}")
                import traceback
                traceback.print_exc()
        
        metrics = target_model.allsplit_epoch_end()
        metrics[f"Val/loss"] = sum([d['loss'] for d in val_loss_list]).item() / len(val_dataloader)
        metrics[f"Val/diff_loss"] = sum([d['diff_loss'] for d in val_loss_list]).item() / len(val_dataloader)
        metrics[f"Val/reward"] = sum([d['reward'] for d in val_loss_list]).item() / len(val_dataloader)
        max_val_rp1 = metrics['Metrics/R_precision_top_1']
        min_val_fid = metrics['Metrics/FID']
        print_table(f'Validation@Step-{global_step}', metrics)
        for mk, mv in metrics.items():
            mk = mk + '_EMA' if ema else mk
            if cfg.vis == "tb":
                writer.add_scalar(mk, mv, global_step=global_step)
            elif cfg.vis == "swanlab":
                writer.log({mk: mv}, step=global_step)
        target_model.denoiser.train()
        return max_val_rp1, min_val_fid, metrics

    max_rp1, min_fid = 0.80862, 0.40636#validation(model)
    # max_rp1, min_fid = validation(model)
    # exit(0)
    #max_rp1, min_fid = validation(model)

    progress_bar = tqdm(range(0, cfg.TRAIN.max_ft_steps), desc="Steps")
    
    # 使用 MotionRewardAdapter 替代原来的 SPM
    # 支持绝对路径和相对路径
    reward_ckpt_path = ft_config['reward_model_ckpt']
    t5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deps', 'sentence-t5-large')
    
    reward_model = MotionRewardAdapter(
        ckpt_path=reward_ckpt_path,
        t5_path=t5_path,
        repr_type='263',  # 支持: '263', '22x3', '135'
        model_size=ft_config.get('reward_model_size', 'base'),  # 通过 model_size 控制配置
        device='cuda'
    ).cuda()
    
    # ==================== 评估 Reward Model ====================
    @torch.no_grad()
    def evaluate_reward_model(reward_model, dataloader, repr_type='263', batch_size=32):
        """在 finetune 之前评估 reward model 的检索性能"""
        logger.info("="*60)
        logger.info("Evaluating Reward Model before finetuning...")
        logger.info("="*60)
        
        reward_model.eval()
        text_list, text_latents, motion_latents = [], [], []
        
        for batch in tqdm(dataloader, desc="Evaluating Reward Model"):
            batch = move_batch_to_device(batch, device)
            texts = batch['text']
            motions = batch['motion']  # [B, T, D]
            lengths = batch['length']
            
            # 处理文本（可能是列表）
            processed_texts = []
            for text in texts:
                if isinstance(text, list):
                    processed_texts.append(text[0])
                else:
                    processed_texts.append(text)
            
            # 处理长度
            processed_lengths = []
            for length in lengths:
                if isinstance(length, torch.Tensor):
                    processed_lengths.append(length.item())
                else:
                    processed_lengths.append(int(length))
            
            # 获取 embeddings
            t_latent = reward_model.model.get_text_embedding(processed_texts)
            m_latent = reward_model.model.get_motion_embedding(
                motions, 
                processed_lengths, 
                repr_type=repr_type
            )
            
            text_list.extend(processed_texts)
            text_latents.extend(t_latent.cpu().numpy())
            motion_latents.extend(m_latent.cpu().numpy())
        
        # 计算检索指标
        test_result = [text_list, text_latents, motion_latents]
        
        logger.info("\n[Reward Model Retrieval Metrics - BS32]")
        bs32_metrics = calculate_retrieval_metrics_small_batches(test_result, epoch=0, fptr=None)
        
        logger.info("\n[Reward Model Retrieval Metrics - Full]")
        full_metrics = calculate_retrieval_metrics(test_result, epoch=0, fptr=None)
        
        logger.info("="*60)
        logger.info(f"BS32 - R@1: {bs32_metrics['R1']}% | R@5: {bs32_metrics['R5']}% | R@10: {bs32_metrics['R10']}%")
        logger.info(f"Full - R@1: {full_metrics['R1']}% | R@5: {full_metrics['R5']}% | R@10: {full_metrics['R10']}%")
        logger.info("="*60)
        
        return {'bs32': bs32_metrics, 'full': full_metrics}
    
    # 在 finetune 之前评估 reward model
    if cfg.get('eval_reward_model', True):  # 默认开启，可通过配置关闭
        reward_metrics = evaluate_reward_model(reward_model, val_dataloader, repr_type='263')
        logger.info(f"Reward Model Evaluation Complete: {reward_metrics}")
    # ==================== 评估 Reward Model 结束 ====================
    
    model.reward_model = reward_model
    # ================================================================================================================= #
    model.mem_dict = {'Before Forward Finetune': [], 'Before Diffusion Reverse': [], 'Before T5': [], 'After T5': [],}
    model.mem_dict.update({f'After Reverse Timestep {i}': [] for i in range(50)})
    model.mem_dict.update({
        'After Reverse': [], 'Before VAE Decode': [],
        'Afrer VAE Decode/Before Reward func': [], 'After Reward func': [],
        'After Forward Finetune': [], 'Before Backward': [], 'After Backward': [],
    })
    model.ft_config = ft_config
    model.lambda_reward = ft_config['lambda_reward']
    model.reward_record = [[] for _ in range(50)]
    model.trn_reward = []
    # ================================================================================================================= #
    logger.info(f'FineTune Config : {ft_config} ')
    global_step = 0
    # 使用命令行参数中的 ft_epochs
    epochs = cfg.ft_epochs
    logger.info(f"Training for {epochs} epochs")
    #_, _, metrics = validation(model)
    #logger.info(f"Epoch -1 (Before Finetune Val Metrics): {metrics} \n\n")
    rcd_reward = []
    
    # 验证频率：每 100 步验证一次
    validation_steps = 100
    
    # ==================== Step 0 初始验证 ====================
    logger.info("="*60)
    logger.info("Running initial validation at step 0...")
    logger.info("="*60)
    cur_rp1, cur_fid, metrics = validation(model)
    save_path = os.path.join(cfg.output_dir, 'checkpoints', 
                            f"step0-R1-{round(cur_rp1, 3)}-FID-{round(cur_fid, 3)}.ckpt")
    ckpt = dict(state_dict=model.state_dict(), 
                ft_config=model.ft_config, metrics=metrics, 
                reward_record=[model.reward_record, model.trn_reward])
    model.on_save_checkpoint(ckpt)
    torch.save(ckpt, save_path)
    logger.info(f"Step 0: Initial validation complete | R@1={round(cur_rp1, 3)}, FID={round(cur_fid, 3)}")
    logger.info(f"Step 0: Saved to {save_path}")
    # ==================== Step 0 初始验证结束 ====================
    
    device = torch.device("cuda:0")
    for epoch in range(epochs):
        para_cnt, para_sum = fuck(model)
        logger.info(f"Epoch {epoch}: Para Sum: {para_sum} Para Cnt: {para_cnt}\n\n")
        for step, batch in enumerate(train_dataloader):
            # print(batch['text'])
            # exit(0)
            batch = move_batch_to_device(batch, device)
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
            peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            model.mem_dict['Before Forward Finetune'].append(peak_memory)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #  record_shapes=True, 
            #  with_stack=True) as prof:
            #     prof.start()
            #     with record_function("train_step"):
            # xxxx += 1
            # if xxxx % 25 == 0:
            #     cur_rp1, cur_fid, metrics = validation(model)
            #     # print('\n\n',cur_rp1, cur_fid, metrics, '\n\n')
            #     logging.info(f"Epoch: {epoch}, Step: {step}, Validation Metrics: {metrics}\n\n")
            if ft_config['type'] == 'NIPS':
                loss_dict = model.allsplit_step('finetune_nips', batch, optimizer, lr_scheduler)
            else:
                loss_dict = model.allsplit_step('finetune', batch)
            if ft_config['type'] != 'NIPS':
                peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
                model.mem_dict['After Forward Finetune'].append(peak_memory)
                #router_loss = loss_dict['router_loss']
                torch.cuda.reset_peak_memory_stats(device)
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                model.mem_dict['Before Backward'].append(peak_memory)
                loss = loss_dict['loss']
                loss.backward()
                device = torch.device("cuda:0")
                torch.cuda.reset_peak_memory_stats(device)
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                model.mem_dict['After Backward'].append(peak_memory)
                torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), cfg.TRAIN.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                # prof.stop()
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    # exit(0)
            diff_loss = loss_dict['diff_loss']
            reward = loss_dict['reward']
            rcd_reward.append(reward.item())
            loss = loss_dict['loss']
            progress_bar.update(1)
            global_step += 1
            logs = {
                'Epoch': epoch,
                #'Gstep': global_step,
                "loss": loss.item(),
                "diff_loss": 0,
                #"router_loss": router_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "reward":reward.item()}
            progress_bar.set_postfix(**logs)
            for k, v in logs.items():
                if cfg.vis == "tb":
                    writer.add_scalar(f"ft/{k}", v, global_step=global_step)
                elif cfg.vis == "swanlab":
                    writer.log({f"ft/{k}": v}, step=global_step)
            
            # 每 100 步验证一次
            if global_step % validation_steps == 0:
                cur_rp1, cur_fid, metrics = validation(model)
                save_path = os.path.join(cfg.output_dir, 'checkpoints', 
                                        f"step{global_step}-R1-{round(cur_rp1, 3)}-FID-{round(cur_fid, 3)}.ckpt")
                ckpt = dict(state_dict=model.state_dict(), 
                            ft_config=model.ft_config, metrics=metrics, 
                            reward_record=[model.reward_record, model.trn_reward])
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                main_metrics = {k: round(v, 4) for k, v in metrics.items() if 'gt' not in k}
                logger.info(f"Step {global_step}: Saved to {save_path} | R@1={round(cur_rp1, 3)}, FID={round(cur_fid, 3)}")
                logger.info(f"Step {global_step}: Metrics={main_metrics}\n")
################################################################################################
        avg_mem_dict = {k: 0 for k in model.mem_dict.keys()}
        
        # for k, v in model.mem_dict.items():
        #    try:
        #        #avg_mem_dict[k] = min(v)#sum(v) / len(v)
        #        avg_mem_dict[k] = sum(v) / len(v)
        #    except:
        #        #avg_mem_dict[k] = min([u[0] for u in v])#sum([u[0]for u in v]) / len([u[0] for u in v])
        #        avg_mem_dict[k] = sum([u[0]for u in v]) / len([u[0] for u in v])
        # a = 1
#         import pickle
# #        with open(f"DRTune-10-5-50.pkl", 'wb') as f:
#             # pickle.dump(rcd_reward, f)
#            #exit(0)
#         with open(f"MemUse_ReFL_20.pkl", 'wb') as f:
#             pickle.dump([avg_mem_dict, model.mem_dict], f)
#             exit(0)
        cur_rp1, cur_fid, metrics = validation(model)
        save_path = os.path.join(cfg.output_dir, 'checkpoints', f"E{epoch}-R1-{round(cur_rp1, 3)}-FID-{round(cur_fid, 3)}.ckpt")
        ckpt = dict(state_dict=model.state_dict(), mem_use=[model.mem_dict, avg_mem_dict], \
                    ft_config=model.ft_config, metrics=metrics, reward_record=[model.reward_record, model.trn_reward])
        model.on_save_checkpoint(ckpt)
        torch.save(ckpt, save_path)
        main_metrics = {k: round(v,4) for k, v in metrics.items() if 'gt' not in k}
        logger.info(f"Epoch: {epoch}, Saved state to {save_path} with R@1:{round(cur_rp1, 3)}, FID:{round(cur_fid, 3)} Metrics: {metrics}")
        logger.info(f"Epoch: {epoch}, Main Metrics: {main_metrics}\n\n")
        # if cur_fid > 2 and cur_rp1 < 0.4:
        #    exit(0)



if __name__ == "__main__":
    main()
