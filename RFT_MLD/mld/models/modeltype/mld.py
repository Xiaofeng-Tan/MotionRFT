import time, random
import inspect
import logging
from typing import Optional
import contextlib
import tqdm
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.temos_utils import lengths_to_mask, remove_padding
from mld.utils.utils import count_parameters, get_guidance_scale_embedding, extract_into_tensor, control_loss_calculate
from mld.data.humanml.utils.plot_script import plot_3d_motion

from .base import BaseModel

# 使用新的 reward adapter 中的函数
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from reward_adapter import process_T5_outputs
from ft_config import ft_reset

logger = logging.getLogger(__name__)


class MLD(BaseModel):
    def __init__(self, cfg: DictConfig, datamodule: BaseDataModule) -> None:
        super().__init__()
        self.reward_model = None
        self.cfg = cfg
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.datamodule = datamodule

        if cfg.model.guidance_scale == 'dynamic':
            s_cfg = cfg.model.scheduler
            self.guidance_scale = s_cfg.cfg_step_map[s_cfg.num_inference_steps]
            logger.info(f'Guidance Scale set as {self.guidance_scale}')

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints

        self.vae_scale_factor = cfg.model.get("vae_scale_factor", 1.0)
        self.guidance_uncondp = cfg.model.get('guidance_uncondp', 0.0)

        logger.info(f"vae_scale_factor: {self.vae_scale_factor}")
        logger.info(f"prediction_type: {self.scheduler.config.prediction_type}")
        logger.info(f"guidance_scale: {self.guidance_scale}")
        logger.info(f"guidance_uncondp: {self.guidance_uncondp}")
        
        self.is_controlnet = False
        #self.is_controlnet = cfg.model.get('is_controlnet', False)
        #if self.is_controlnet:
        #    c_cfg = self.cfg.model.denoiser.copy()
        #    c_cfg['params']['is_controlnet'] = True
        #    self.controlnet = instantiate_from_config(c_cfg)
        #    self.traj_encoder = instantiate_from_config(cfg.model.traj_encoder)
#
        #    self.vaeloss = cfg.model.get('vaeloss', False)
        #    self.vaeloss_type = cfg.model.get('vaeloss_type', 'sum')
        #    self.cond_ratio = cfg.model.get('cond_ratio', 0.0)
        #    self.rot_ratio = cfg.model.get('rot_ratio', 0.0)
        #    self.control_loss_func = cfg.model.get('control_loss_func', 'l2')
        #    if self.vaeloss and self.cond_ratio == 0.0 and self.rot_ratio == 0.0:
        #        raise ValueError("Error: When 'vaeloss' is True, 'cond_ratio' and 'rot_ratio' cannot both be 0.")
        #    self.use_3d = cfg.model.get('use_3d', False)
        #    self.guess_mode = cfg.model.get('guess_mode', False)
        #    if self.guess_mode and not self.do_classifier_free_guidance:
        #        raise ValueError(
        #            "Invalid configuration: 'guess_mode' is enabled, but 'do_classifier_free_guidance' is not. "
        #            "Ensure that 'do_classifier_free_guidance' is True (MLD) when 'guess_mode' is active."
        #        )
        #    self.lcm_w_min_nax = cfg.model.get('lcm_w_min_nax')
        #    self.lcm_num_ddim_timesteps = cfg.model.get('lcm_num_ddim_timesteps')
        #    if (self.lcm_w_min_nax is not None or self.lcm_num_ddim_timesteps is not None) and self.denoiser.time_cond_proj_dim is None:
        #        raise ValueError(
        #            "Invalid configuration: When either 'lcm_w_min_nax' or 'lcm_num_ddim_timesteps' is not None, "
        #            "'denoiser.time_cond_proj_dim' must be None (MotionLCM)."
        #        )
#
        #    logger.info(f"vaeloss: {self.vaeloss}, "
        #                f"vaeloss_type: {self.vaeloss_type}, "
        #                f"cond_ratio: {self.cond_ratio}, "
        #                f"rot_ratio: {self.rot_ratio}, "
        #                f"control_loss_func: {self.control_loss_func}")
        #    logger.info(f"use_3d: {self.use_3d}, "
        #                f"guess_mode: {self.guess_mode}")
        #    logger.info(f"lcm_w_min_nax: {self.lcm_w_min_nax}, "
        #                f"lcm_num_ddim_timesteps: {self.lcm_num_ddim_timesteps}")
#
        #    #time.sleep(2)  # 留个心眼

        #self.dno = instantiate_from_config(cfg.model['noise_optimizer']) \
            #if cfg.model.get('noise_optimizer') else None

        self.summarize_parameters()
        self.gflops = 0
        self.update_cnt = 0
    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1 and self.denoiser.time_cond_proj_dim is None

    def summarize_parameters(self) -> None:
        logger.info(f'VAE Encoder: {count_parameters(self.vae.encoder)}M')
        logger.info(f'VAE Decoder: {count_parameters(self.vae.decoder)}M')
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')


    def forward(self, batch: dict, add_noise=None) -> tuple:
        texts = batch["text"]
        feats_ref = batch.get("motion")
        lengths = batch["length"]
        # hint = batch.get('hint')
        # hint_mask = batch.get('hint_mask')
        mask = batch["mask"]
        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)

        # text_emb = self.text_encoder(texts)
        t_len, token_embeddings, text_emb = process_T5_outputs(texts, self.text_encoder.text_model)
        controlnet_cond = None
        # if self.is_controlnet:
        #     assert hint is not None
        #     hint_reshaped = hint.view(hint.shape[0], hint.shape[1], -1)
        #     hint_mask_reshaped = hint_mask.view(hint_mask.shape[0], hint_mask.shape[1], -1).sum(dim=-1) != 0
        #     controlnet_cond = self.traj_encoder(hint_reshaped, hint_mask_reshaped)

        # latents = torch.randn((len(lengths), *self.latent_dim), device=text_emb.device)
        feats_ref = feats_ref.cuda()
        mask = lengths_to_mask(lengths, feats_ref.device)
        batch["mask"] = mask
        latents, _ = self.vae.encode(feats_ref, mask)
        if add_noise is not None:
            latents, timestep = add_noise(latents)
            timestep = torch.stack([timestep, timestep], dim = 0).reshape(-1).cuda()
            latents = latents.cuda()
        mask = batch.get('mask', lengths_to_mask(lengths, text_emb.device))


        latents = self._diffusion_reverse(latents, text_emb, controlnet_cond=controlnet_cond, diy_timestep=timestep,batch=batch)
        feats_rst = self.vae.decode(latents / self.vae_scale_factor, mask)
        
        if add_noise is not None:
            return feats_rst
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if feats_ref is not None:
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref

    def predicted_origin(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> tuple:
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")

        return pred_original_sample, pred_epsilon

    def _diffusion_reverse(
            self,
            latents: torch.Tensor,
            cond: torch.Tensor,
            controlnet_cond: Optional[torch.Tensor] = None,
            token_embeddings = None,
            batch = None,
            diy_timestep=None,
    ) -> torch.Tensor:
        
        
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(cond.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        if self.is_controlnet and self.do_classifier_free_guidance and not self.guess_mode:
            controlnet_cond = torch.cat([controlnet_cond] * 2)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            controlnet_residuals = None
            # predict the noise residual
            if diy_timestep is not None:
                with torch.no_grad():
                    model_output = self.denoiser(
                    sample=latent_model_input,
                    timestep=diy_timestep,
                    timestep_cond=None,
                    encoder_hidden_states=cond,
                    controlnet_residuals=None)[0]
            else:
                    with torch.no_grad():
                        model_output = self.denoiser(
                        sample=latent_model_input,
                        timestep=t,
                        timestep_cond=None,
                        encoder_hidden_states=cond,
                        controlnet_residuals=None)[0]
            # model_output require grad True
            # loss = - Reward 
            # 
            # perform guidance
            if self.do_classifier_free_guidance:
                model_output_text, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (model_output_text - model_output_uncond)

            # if diy_timestep is not None:

            #     x_i_0, _ = self.predicted_origin(model_output, diy_timestep[:model_output.shape[0]].long().cuda(), latents)
            #     return x_i_0
            #     motion_x0 = self.vae.decode(x_i_0 / self.vae_scale_factor, batch['mask'])
            #     return motion_x0
            #  self.latents_list_x0.append(x_i_0.detach().cpu().numpy())
            x_i_0, _ = self.predicted_origin(model_output, torch.tensor([t]).long().cuda(), latents)
            # motion_x0 = self.vae.decode(x_i_0 / self.vae_scale_factor, batch['mask'])
            latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample
            # motion_xt = self.vae.decode(latents / self.vae_scale_factor, batch['mask'])
            
            # self.motion_list_x0.append(self.feats2joints(motion_x0).detach().cpu().numpy())
            # self.motion_list_xt.append(self.feats2joints(motion_xt).detach().cpu().numpy())
            
            # self.latents_list_xt.append(latents.detach().cpu().numpy())
            # torch.cuda.empty_cache()
        # with torch.no_grad():
        #    motion_fake = self.vae.decode(x_i_0 / self.vae_scale_factor, batch['mask'])
        #    reward, m_latent = self.reward_model.get_reward_t2m(raw_texts=batch["text"], sent_emb=token_embeddings,t_len=batch['t_len'], \
        #                                               motion_feats=motion_fake, m_len=batch['length'], timestep=t)
            
        #    self.m_latents_list_x0.append(m_latent.detach().cpu().numpy())
        #    torch.cuda.empty_cache()
        #    motion_fake = self.vae.decode(latents / self.vae_scale_factor, batch['mask'])
        #    reward, m_latent = self.reward_model.get_reward_t2m(raw_texts=batch["text"], sent_emb=token_embeddings,t_len=batch['t_len'], \
        #                                               motion_feats=motion_fake, m_len=batch['length'], timestep=t)
            
        #    self.m_latents_list_xt.append(m_latent.detach().cpu().numpy())
            
           # self.reward_record[i].append(reward.item())
            #    torch.cuda.empty_cache()
            
        return latents

    def _diffusion_process(self, latents: torch.Tensor, encoder_hidden_states: torch.Tensor):

        #controlnet_cond = None
        #if self.is_controlnet:
        #    assert hint is not None
        #    hint_reshaped = hint.view(hint.shape[0], hint.shape[1], -1)
        #    hint_mask_reshaped = hint_mask.view(hint_mask.shape[0], hint_mask.shape[1], -1).sum(-1) != 0
        #    controlnet_cond = self.traj_encoder(hint_reshaped, mask=hint_mask_reshaped)

        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        #if self.denoiser.time_cond_proj_dim is not None and self.lcm_num_ddim_timesteps is not None:
        #    step_size = self.scheduler.config.num_train_timesteps  self.lcm_num_ddim_timesteps
        #    candidate_timesteps = torch.arange(
        #        start=step_size - 1,
        #        end=self.scheduler.config.num_train_timesteps,
        #        step=step_size,
        #        device=latents.device
        #    )
        #    timesteps = candidate_timesteps[torch.randint(
        #        low=0,
        #        high=candidate_timesteps.size(0),
        #        size=(bsz,),
        #        device=latents.device
        #    )]
        #else:
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        )
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        timestep_cond = None
        #if self.denoiser.time_cond_proj_dim is not None:
        #    if self.lcm_w_min_nax is None:
        #        w = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
        #    else:
        #        w = (self.lcm_w_min_nax[1] - self.lcm_w_min_nax[0]) * torch.rand((bsz,)) + self.lcm_w_min_nax[0]
        #    timestep_cond = get_guidance_scale_embedding(
        #        w, embedding_dim=self.denoiser.time_cond_proj_dim
        #    ).to(device=latents.device, dtype=latents.dtype)

        controlnet_residuals = None
        router_loss_controlnet = None
        #if self.is_controlnet:
        #    controlnet_residuals, router_loss_controlnet = self.controlnet(
        #        sample=noisy_latents,
        #        timestep=timesteps,
        #        timestep_cond=timestep_cond,
        #        encoder_hidden_states=encoder_hidden_states,
        #        controlnet_cond=controlnet_cond)

        model_output, router_loss = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            timestep_cond=timestep_cond,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_residuals=controlnet_residuals)

        latents_pred, noise_pred = self.predicted_origin(model_output, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "sample_pred": latents_pred,
            "sample_gt": latents,
            "router_loss": router_loss_controlnet if self.is_controlnet else router_loss
        }
        return n_set
    
    def train_diffusion_forward(self, batch: dict) -> dict:
        feats_ref = batch["motion"]
        mask = batch['mask']
        hint = batch.get('hint', None)
        hint_mask = batch.get('hint_mask', None)

        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, mask)
            z = z * self.vae_scale_factor

        text = batch["text"]
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]
        text_emb = self.text_encoder(text)
        n_set = self._diffusion_process(z, text_emb)#, hint=hint, hint_mask=hint_mask)

        loss_dict = dict()

        if self.denoiser.time_cond_proj_dim is not None:
            pass
            ## LCM (only used in motion ControlNet)
            #model_pred, target = n_set['sample_pred'], n_set['sample_gt']
            ## Performance comparison: l2 loss > huber loss when training controlnet for LCM
            #diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # DM
            if self.scheduler.config.prediction_type == "epsilon":
                model_pred, target = n_set['noise_pred'], n_set['noise'] # 
            elif self.scheduler.config.prediction_type == "sample":
                pass #model_pred, target = n_set['sample_pred'], n_set['sample_gt']
            else:
                raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")
            
            reward = torch.tensor(0., device=model_pred.device)

            diff_loss = F.mse_loss(model_pred, target, reduction="mean") # wengwanjiang
        
        loss_dict['diff_loss'] = diff_loss #rw_grad * self.lambda_reward
        loss_dict['reward'] = reward
        # Router loss
        loss_dict['router_loss'] = n_set['router_loss'] if n_set['router_loss'] is not None \
            else torch.tensor(0., device=diff_loss.device)

        if self.is_controlnet and self.vaeloss:
            feats_rst = self.vae.decode(n_set['sample_pred'] / self.vae_scale_factor, mask)

            if self.cond_ratio != 0:
                joints_rst = self.feats2joints(feats_rst)
                if self.use_3d:
                    hint = self.datamodule.denorm_spatial(hint)
                else:
                    joints_rst = self.datamodule.norm_spatial(joints_rst)
                hint_mask = hint_mask.sum(-1, keepdim=True) != 0
                cond_loss = control_loss_calculate(self.vaeloss_type, self.control_loss_func, joints_rst, hint,
                                                   hint_mask)
                loss_dict['cond_loss'] = self.cond_ratio * cond_loss
            else:
                loss_dict['cond_loss'] = torch.tensor(0., device=diff_loss.device)

            if self.rot_ratio != 0:
                mask = mask.unsqueeze(-1)
                rot_loss = control_loss_calculate(self.vaeloss_type, self.control_loss_func, feats_rst, feats_ref, mask)
                loss_dict['rot_loss'] = self.rot_ratio * rot_loss
            else:
                loss_dict['rot_loss'] = torch.tensor(0., device=diff_loss.device)

        else:
            loss_dict['cond_loss'] = loss_dict['rot_loss'] = torch.tensor(0., device=diff_loss.device)

        total_loss = sum(loss_dict.values())
        #total_loss = sum(value for key, value in loss_dict.items() if key != 'reward')
        loss_dict['loss'] = total_loss
        return loss_dict
    

    def _diffusion_reverse_ft(self, x_i, cond, strategy):
        
        # scale the initial noise by the standard deviation required by the scheduler
        x_i = x_i * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(cond.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
        # torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        # self.mem_dict['Before Diffusion Reverse'].append(peak_memory)
        #print(f"Before Diffusion Reverse GPU Usage: {peak_memory:.2f} MB")
        
        for i, t in enumerate(timesteps):

            # expand the latents if we are doing classifier free guidance
            if 'DRTune' == self.ft_config['type']:
                latent_model_input = (torch.cat([x_i.detach()] * 2) if self.do_classifier_free_guidance else x_i.detach())
                latent_model_input = latent_model_input.detach()
            else:
                latent_model_input = (torch.cat([x_i] * 2) if self.do_classifier_free_guidance else x_i)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            enable_grad = i in self.ft_config['enable_grad']
            #print(i, '==', enable_grad)
            ctx = torch.enable_grad() if enable_grad else torch.no_grad() # 
            # predict the noise residual
            with ctx:
                if 'DRTune' == self.ft_config['type'] or \
                    ('DRaFT' == self.ft_config['type'] and i not in self.ft_config['t_train']):
                    model_output = self.denoiser(sample=latent_model_input, timestep=t, encoder_hidden_states=cond)[0]
                    if i not in self.ft_config['t_train']:
                        model_output = model_output.detach()
                else:
                    model_output = self.denoiser(sample=latent_model_input, timestep=t, encoder_hidden_states=cond)[0]
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict[f'After Reverse Timestep {i}'].append([peak_memory, 1 if enable_grad else 0])
            #torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
            #peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            #torch.cuda.empty_cache()
            if self.do_classifier_free_guidance:
                model_output_text, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (model_output_text - model_output_uncond)

            if ('ReFL' == self.ft_config['type'] and i == self.ft_config['t_min']) or \
                ('DRTune' == self.ft_config['type'] and i == self.ft_config['t_min']):
                batched_t = torch.tensor(t.item(), dtype=torch.long).unsqueeze(0).expand(x_i.shape[0]).to(x_i.device)
                x_i, _ = self.predicted_origin(model_output, batched_t, x_i) # x_0
                return x_i
            
            x_i = self.scheduler.step(model_output, t, x_i, **extra_step_kwargs).prev_sample

        return x_i
    
    def ft_diffusion_forward(self, batch: dict) -> dict:
        feats_ref = batch["motion"]
        mask = batch['mask']
        
        text = batch["text"]
        if self.do_classifier_free_guidance:
            text = text + [""] * len(text)
        
        
        # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
        # self.mem_dict['Before T5'].append(peak_memory)
        t_len, token_embeddings, text_emb = process_T5_outputs(text, self.text_encoder.text_model)
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, mask)
            z = z * self.vae_scale_factor
        # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
        # self.mem_dict['After T5'].append(peak_memory)
        # torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        x_T = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)
        strategy = None
        x_0 = self._diffusion_reverse_ft(x_T, text_emb, strategy)
        if self.ft_config.get('type') not in [None, 'None']:
            self.ft_config = ft_reset(self.ft_config)
        loss_dict = dict()
        # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
        # self.mem_dict['After Reverse'].append(peak_memory)
        reward = torch.tensor(0., device=text_emb.device)
        if self.reward_model is not None and self.lambda_reward != 0:
            x_0 = x_0 / self.vae_scale_factor
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict['Before VAE Decode'].append(peak_memory)
            recons_motion = self.vae.decode(x_0, mask)
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            t_len = t_len[:len(t_len)//2]
            token_embeddings = token_embeddings[:token_embeddings.shape[0]//2]
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict['Afrer VAE Decode/Before Reward func'].append(peak_memory)
            reward = 0 - self.reward_model.get_reward_t2m(raw_texts=batch["text"], motion_feats=recons_motion, m_len=batch["length"], \
                                                           t_len=t_len, sent_emb=token_embeddings,timestep=torch.tensor(0, dtype=torch.long).to(x_0.device))
            # reward = torch.tensor(0.0).cuda()
            self.trn_reward.append(reward.mean().item())
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict['After Reward func'].append(peak_memory)

            reward = reward * self.lambda_reward
        zee = torch.rand_like(x_0).cuda()
        diff_loss = F.mse_loss(x_0, z, reduction="mean") # wengwanjiang

        loss_dict['diff_loss'] = torch.tensor(0., device=text_emb.device) 
        # loss_dict['reward'] = F.mse_loss(x_0, z, reduction="mean") # wengwanjiang
        
        # reward 需要取平均值变成标量
        loss_dict['reward'] = reward.mean() if reward.dim() > 0 else reward
        
        total_loss = sum(loss_dict.values())
        loss_dict['loss'] = total_loss
        return loss_dict
    
    
    
    def generate_noisy_motion(self, batch):
        return 1
    
    def t2m_eval(self, batch: dict) -> dict:
        
        texts = batch["text"]
        feats_ref = batch["motion"]
        mask = batch['mask']
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]
        hint = batch.get('hint', None)
        hint_mask = batch.get('hint_mask', None)

        start = time.time()

        if self.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            mask = mask.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            hint = hint and hint.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            hint_mask = hint_mask and hint_mask.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)

        text_st = time.time()
        # text_emb = self.text_encoder(texts)
        t_len, token_embeddings, text_emb = process_T5_outputs(texts, self.text_encoder.text_model)
        batch['t_len'] = t_len[:len(t_len)//2]
        token_embeddings = token_embeddings[:len(t_len)//2]
        text_et = time.time()
        self.text_encoder_times.append(text_et - text_st)

        controlnet_cond = None
        # if self.is_controlnet:
        #     assert hint is not None
        #     hint_st = time.time()
        #     hint_reshaped = hint.view(hint.shape[0], hint.shape[1], -1)
        #     hint_mask_reshaped = hint_mask.view(hint_mask.shape[0], hint_mask.shape[1], -1).sum(dim=-1) != 0
        #     controlnet_cond = self.traj_encoder(hint_reshaped, hint_mask_reshaped)
        #     hint_et = time.time()
        #     self.traj_encoder_times.append(hint_et - hint_st)

        diff_st = time.time()

        latents = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)
        # self.latents_list_xt = []
        # self.latents_list_x0 = []
        # self.m_latents_list_xt = []
        # self.m_latents_list_x0 = []
        
        # 正常验证流程
        latents = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)
        latents = self._diffusion_reverse(latents, text_emb, controlnet_cond=None, token_embeddings=token_embeddings, batch=batch)
        # print(self.motion_list_xt.shape, batch["text"], batch["length"])
        # exit(0)
        # print(self.latents_list_x0[0].shape, len(self.latents_list_x0))
        # print(self.m_latents_list_x0[0].shape, len(self.m_latents_list_x0))               
        
        # print(self.latents_list_xt[0].shape, len(self.latents_list_xt))
        # print(self.m_latents_list_xt[0].shape, len(self.m_latents_list_xt))     

        # import pickle 
        # with open('iclr_rebuttal/eztune_mld_latent_16step.pkl', 'wb') as f:
        #     pickle.dump({"raw_latents": self.latents_list_x0, \
        #                 "reward_latents": self.m_latents_list_x0,
        #                 "raw_latents_xt": self.latents_list_xt,
        #                 "reward_latents_xt": self.m_latents_list_xt,
        #         }, f)
            
        # exit(0)
        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        vae_st = time.time()
        feats_rst = self.vae.decode(latents / self.vae_scale_factor, mask)
        feats_rst_raw = feats_rst.clone()  # 保存原始特征（未经 renorm）
        # for i in range(bs):
        #     # 确定当前索引属于哪个段落(0,1,2,3)
        #     batch["text"][i] = ""
        # with torch.no_grad():
        #     reward_p = self.reward_model.get_reward_t2m(raw_texts=batch["text"], sent_emb=token_embeddings,t_len=batch['t_len'], \
        #                                               motion_feats=feats_rst, m_len=batch['length'], timestep=None)
        #     reward_g = self.reward_model.get_reward_t2m(raw_texts=batch["text"], sent_emb=token_embeddings,t_len=batch['t_len'], \
        #                                               motion_feats=feats_ref, m_len=batch['length'], timestep=None)
        # print(reward_p > reward_g)
        # exit(0)
        vae_et = time.time()
        self.vae_decode_times.append(vae_et - vae_st)

        self.frames.extend(lengths)

        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, eval(f"self.cfg.DATASET.{self.cfg.DATASET.NAME.upper()}.UNIT_LEN"),
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {"m_ref": feats_ref, "m_rst": feats_rst,
                  "lat_t": text_emb, "lat_m": motion_emb, "lat_rm": recons_emb,
                  "joints_ref": joints_ref, "joints_rst": joints_rst,
                  "feats_rst_raw": feats_rst_raw}

        if 'hint' in batch:
            hint_3d = self.datamodule.denorm_spatial(batch['hint']) * batch['hint_mask']
            rs_set['hint'] = hint_3d
            rs_set['hint_mask'] = batch['hint_mask']

        return rs_set

    def allsplit_step(self, split: str, batch: dict, optimizer=None, lr_scheduler=None, policy_model=None) -> Optional[dict]:
        if split in ["test", "val"]:
            rs_set = self.t2m_eval(batch)

            if self.datamodule.is_mm:
                metric_list = ['MMMetrics']
            else:
                metric_list = self.metric_list

            for metric in metric_list:
                if metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"])
                elif metric == "MMMetrics" and self.datamodule.is_mm:
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0), batch["length"])
                elif metric == 'ControlMetrics':
                    getattr(self, metric).update(rs_set["joints_rst"], rs_set['hint'],
                                                 rs_set['hint_mask'], batch['length'])
                else:
                    raise TypeError(f"Not support this metric: {metric}.")

        if split in ["train", "val"]:
            loss_dict = self.train_diffusion_forward(batch)
            return loss_dict
        elif split in ["finetune"]:
            loss_dict = self.ft_diffusion_forward(batch)
            return loss_dict
        elif split in ["finetune_nips"]:
            loss_dict = self.ft_diffusion_forward_nips(batch, optimizer, lr_scheduler, policy_model=policy_model)
            return loss_dict
        
    def ft_diffusion_forward_nips(self, batch: dict, optimizer, lr_scheduler, policy_model=None) -> dict:
        feats_ref = batch["motion"]
        mask = batch['mask']
        text = batch["text"]
        if self.do_classifier_free_guidance:
            text = text + [""] * len(text)

        # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
        # self.mem_dict['Before T5'].append(peak_memory)
        t_len, token_embeddings, text_emb = process_T5_outputs(text, self.text_encoder.text_model)
        # with torch.no_grad():
        #     z, dist = self.vae.encode(feats_ref, mask)
        #     z = z * self.vae_scale_factor
        # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
        # self.mem_dict['After T5'].append(peak_memory)
        # torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        
        x_T = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)
        x_0, reward = self._diffusion_reverse_ft_nips(x_T, text_emb, optimizer, batch, t_len[:len(t_len)//2], token_embeddings[:len(t_len)//2], \
                    lr_scheduler, policy_model=policy_model)
        # x_i, cond, optimizer, batch, t_len, token_embeddings, lr_scheduler
        # self.ft_config = ft_reset(self.ft_config)
        # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
        # self.mem_dict['After Reverse'].append(peak_memory)
        loss_dict = dict()
        loss_dict['diff_loss'] = torch.tensor(0., device=text_emb.device) 
        loss_dict['reward'] = reward #torch.tensor(0., device=text_emb.device) 

        total_loss = sum(loss_dict.values())
        loss_dict['loss'] = total_loss
        return loss_dict
    
    def _diffusion_reverse_ft_nips(self, x_i, cond, optimizer, batch, t_len, token_embeddings, lr_scheduler, policy_model=None):
        
        # ==================== Motion Reward Timestep Scheduling ====================
        # 如果开启了 curriculum 模式，根据训练进度非均匀地更新 enable_grad 窗口
        # 早期(高噪声)少优化，后期(低噪声)多优化
        if self.ft_config.get('curriculum', False):
            from ft_config import motion_reward_curriculum_update
            _progress = getattr(self, 'training_progress', 0.0)  # 由外部训练循环设置
            motion_reward_curriculum_update(self.ft_config, _progress, total_steps=self.cfg.model.scheduler.num_inference_steps)
        
        # scale the initial noise by the standard deviation required by the scheduler
        x_i = x_i * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(cond.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
        # torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        # self.mem_dict['Before Diffusion Reverse'].append(peak_memory)
        #print(f"Before Diffusion Reverse GPU Usage: {peak_memory:.2f} MB")
        
        for i, t in enumerate(timesteps):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat([x_i] * 2) if self.do_classifier_free_guidance else x_i)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            enable_grad = i in self.ft_config['enable_grad']
            #print(i, '==', enable_grad)
            ctx = torch.enable_grad() if enable_grad else torch.no_grad() # 
            # predict the noise residual
            with ctx:
                model_output = self.denoiser(sample=latent_model_input, timestep=t, encoder_hidden_states=cond)[0]
            
            x_i_0, _ = self.predicted_origin(model_output, torch.tensor([t]).long().cuda(), latent_model_input)
            cond_x_i_0, uncond_x_i_0 = x_i_0.chunk(2)
            x_i_0 = uncond_x_i_0 + self.guidance_scale * (cond_x_i_0 - uncond_x_i_0)
            
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict[f'After Reverse Timestep {i}'].append([peak_memory, 1 if enable_grad else 0])
            if self.do_classifier_free_guidance:
                model_output_text, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (model_output_text - model_output_uncond)
            
            x_i = self.scheduler.step(model_output, t, x_i, **extra_step_kwargs).prev_sample
            
            
            if enable_grad == False:
                continue
            # prex = x_i.sum()
            #x_i_i = x_i.clone() / self.vae_scale_factor
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict['Before VAE Decode'].append(peak_memory)
            
            recons_motion = self.vae.decode(x_i_0 / self.vae_scale_factor, batch['mask'])
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict['Afrer VAE Decode/Before Reward func'].append(peak_memory)
            
            # ==================== 可视化 recons_motion (输入到 Critic 的 motion) ====================
            # 检查是否禁用调试可视化
            _enable_debug_vis = not getattr(self.cfg, 'no_debug_vis', False)
            
            if _enable_debug_vis:
                # 与 vis_scripts/visualize_263.py 完全一致的可视化流程
                _vis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'debug_vis_critic_input')
                os.makedirs(_vis_dir, exist_ok=True)
                print(f"\n{'='*70}")
                print(f"[DEBUG VIS] 可视化输入到 Critic 的 motion")
                print(f"[DEBUG VIS] recons_motion shape: {recons_motion.shape}, range: [{recons_motion.min().item():.4f}, {recons_motion.max().item():.4f}]")
                print(f"[DEBUG VIS] batch['motion'] shape: {batch['motion'].shape}, range: [{batch['motion'].min().item():.4f}, {batch['motion'].max().item():.4f}]")
                print(f"[DEBUG VIS] 保存目录: {_vis_dir}")
                print(f"{'='*70}")
                
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib.animation import FuncAnimation
                import mpl_toolkits.mplot3d.axes3d as p3
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                from textwrap import wrap
                from mld.data.humanml.scripts.motion_process import recover_from_ric as _vis_recover_from_ric
                
                _T2M_KINEMATIC_CHAIN = [
                    [0, 2, 5, 8, 11],
                    [0, 1, 4, 7, 10],
                    [0, 3, 6, 9, 12, 15],
                    [9, 14, 17, 19, 21],
                    [9, 13, 16, 18, 20],
                ]
                
                def _vis_plot_motion_animation(joints, kinematic_chain, save_path, title="Motion Animation", fps=20, radius=3):
                    """与 vis_scripts/visualize_263.py 中 plot_motion_animation 完全一致"""
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
                        for ci, (chain, color) in enumerate(zip(kinematic_chain, colors)):
                            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2],
                                      linewidth=4.0 if ci < 5 else 2.0, color=color)
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
                    print(f"  动画已保存到: {save_path}")
                    plt.close()
                
                def _vis_plot_motion_sequence(joints, kinematic_chain, save_path, num_frames=8, title="Motion Sequence"):
                    """与 vis_scripts/visualize_263.py 中 plot_motion_sequence 完全一致"""
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
                    print(f"  帧序列图已保存到: {save_path}")
                    plt.close()
                
                # 获取 mean/std 用于反归一化
                _mean = torch.tensor(self.datamodule.hparams['mean']).to(recons_motion)
                _std = torch.tensor(self.datamodule.hparams['std']).to(recons_motion)
                
                _num_vis = min(4, recons_motion.shape[0])  # 最多可视化 4 个样本
                _lengths = batch["length"]
                if isinstance(_lengths, torch.Tensor):
                    _lengths = _lengths.tolist()
                
                for _si in range(_num_vis):
                    _len = int(_lengths[_si])
                    _text = batch["text"][_si]
                    if isinstance(_text, list):
                        _text = _text[0]
                    _safe_text = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in _text[:40]).replace(' ', '_')
                    
                    _gen_znorm = recons_motion[_si, :_len].detach().cpu()  # (T, 263)
                    print(f"  [Sample {_si}] text: '{_text[:60]}', length: {_len}")
                    print(f"  [Sample {_si}] gen (Z-norm) range: [{_gen_znorm.numpy().min():.4f}, {_gen_znorm.numpy().max():.4f}]")
                    
                    # 反归一化后可视化 (真实物理空间) - 与 visualize_263.py 一模一样
                    _gen_denorm = _gen_znorm * _std.cpu() + _mean.cpu()
                    _gen_denorm_np = _gen_denorm.numpy()
                    np.save(os.path.join(_vis_dir, f'sample{_si}_gen_denorm_263.npy'), _gen_denorm_np)
                    print(f"  [Sample {_si}] gen (denorm) range: [{_gen_denorm_np.min():.4f}, {_gen_denorm_np.max():.4f}]")
                    
                    _gen_denorm_joints = _vis_recover_from_ric(_gen_denorm.unsqueeze(0).float(), joints_num=22)
                    _gen_denorm_joints = _gen_denorm_joints.squeeze(0).numpy()
                    print(f"  [Sample {_si}] gen (denorm) joints range: [{_gen_denorm_joints.min():.4f}, {_gen_denorm_joints.max():.4f}]")
                    _vis_plot_motion_animation(_gen_denorm_joints, _T2M_KINEMATIC_CHAIN,
                        save_path=os.path.join(_vis_dir, f'sample{_si}_gen_denorm_{_safe_text}.gif'),
                        title=f"Gen Denorm (real space): {_text[:50]}", fps=20)
                    _vis_plot_motion_sequence(_gen_denorm_joints, _T2M_KINEMATIC_CHAIN,
                        save_path=os.path.join(_vis_dir, f'sample{_si}_gen_denorm_{_safe_text}_seq.png'),
                        title=f"Gen Denorm: {_text[:50]}")
                    
                    # GT motion 反归一化后可视化
                    _gt_znorm = batch["motion"][_si, :_len].detach().cpu()
                    print(f"  [Sample {_si}] GT (Z-norm) range: [{_gt_znorm.numpy().min():.4f}, {_gt_znorm.numpy().max():.4f}]")
                    
                    _gt_denorm = _gt_znorm * _std.cpu() + _mean.cpu()
                    _gt_denorm_np = _gt_denorm.numpy()
                    np.save(os.path.join(_vis_dir, f'sample{_si}_gt_denorm_263.npy'), _gt_denorm_np)
                    
                    _gt_denorm_joints = _vis_recover_from_ric(_gt_denorm.unsqueeze(0).float(), joints_num=22)
                    _gt_denorm_joints = _gt_denorm_joints.squeeze(0).numpy()
                    print(f"  [Sample {_si}] GT (denorm) joints range: [{_gt_denorm_joints.min():.4f}, {_gt_denorm_joints.max():.4f}]")
                    _vis_plot_motion_animation(_gt_denorm_joints, _T2M_KINEMATIC_CHAIN,
                        save_path=os.path.join(_vis_dir, f'sample{_si}_gt_denorm_{_safe_text}.gif'),
                        title=f"GT Denorm (real space): {_text[:50]}", fps=20)
                    _vis_plot_motion_sequence(_gt_denorm_joints, _T2M_KINEMATIC_CHAIN,
                        save_path=os.path.join(_vis_dir, f'sample{_si}_gt_denorm_{_safe_text}_seq.png'),
                        title=f"GT Denorm: {_text[:50]}")
                    
                    print()
                
                print(f"{'='*70}")
                print(f"[DEBUG VIS] 可视化完成! 共 {_num_vis} 个样本")
                print(f"[DEBUG VIS] 每个样本包含:")
                print(f"  - gen_denorm: 生成 motion 反归一化后 (真实物理空间, 与 visualize_263.py 一致)")
                print(f"  - gt_denorm: GT motion 反归一化后 (真实物理空间)")
                print(f"  - .npy 文件: 可用 visualize_263.py 直接打开")
                print(f"[DEBUG VIS] 保存目录: {_vis_dir}")
                print(f"{'='*70}")
                
                # 停止训练
                import sys as _sys
                print("\n[DEBUG VIS] 可视化完成，停止训练流程。")
                _sys.exit(0)
            # ==================== 可视化结束 ====================
            
            # ==================== 两段式 Reward 策略 ====================
            # i < reward_t_switch (高噪声段): 用 predicted x_0, timestep=0 → R(x_0, 0)
            # i >= reward_t_switch (低噪声段): 直接用 x_t, timestep=t → R(x_t, t)
            _reward_t_switch = getattr(self, 'reward_t_switch', 50)
            
            if i < _reward_t_switch:
                # 高噪声段：用单步预测的 x_0（已有的 x_i_0），以 timestep=0 计算 reward
                reward_input_motion = recons_motion  # 已经是 decode(x_i_0) 的结果
                reward_timestep = torch.tensor(0, dtype=torch.long).to(x_i.device)
            else:
                # 低噪声段：直接用 x_t（当前 x_i），以 timestep=t 计算 reward
                reward_input_motion = self.vae.decode(x_i / self.vae_scale_factor, batch['mask'])
                reward_timestep = torch.clamp(t, max=getattr(self, 'reward_max_t', 500) - 1).long().to(x_i.device)
            
            reward = self.reward_model.get_reward_t2m(
                raw_texts=batch["text"], 
                motion_feats=reward_input_motion, 
                m_len=batch["length"],
                t_len=t_len, 
                sent_emb=token_embeddings,
                timestep=reward_timestep,
                gt_motion_feats=batch["motion"],  # GT motion for m2m reward
                gt_len=batch["length"]            # GT motion lengths
            )
            self.trn_reward.append(reward.mean().item())
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict['After Reward func'].append(peak_memory)

            reward = 0. - reward.mean() * self.lambda_reward
            # if self.ft_config['dy'] == 0:
            #     reward *= (1 - i / len(timesteps))
            # elif self.ft_config['dy'] == 1:
            #     reward *= (i / len(timesteps))
            # peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            # self.mem_dict['After Forward Finetune'].append(peak_memory)

            diff_loss = torch.tensor(0).to(x_i.device)
            loss = diff_loss + reward
            #router_loss = loss_dict['router_loss']
            # device = torch.device("cuda:0")
            # torch.cuda.reset_peak_memory_stats(device)
            # peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            # self.mem_dict['Before Backward'].append(peak_memory)
            # self.update_cnt += 1
            # if self.update_cnt % 25 == 0:
                
            if policy_model is None:
                loss.backward()#(retain_graph=True)
            else:
                loss.backward()
                for pA, pB in zip(policy_model.denoiser.parameters(), self.denoiser.parameters()):
                    pA.grad = pB.grad.clone()
                    pB.grad = None
                
            # device = torch.device("cuda:0")
            # torch.cuda.reset_peak_memory_stats(device)
            # peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            # self.mem_dict['After Backward'].append(peak_memory)
            if policy_model is None:
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.cfg.TRAIN.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(policy_model.denoiser.parameters(), self.cfg.TRAIN.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            # cnt1, cnt2 = 0, 0
            # sum1, sum2 = 0, 0
            # for k, v in self.denoiser.state_dict().items():
            #     cnt1 += v.numel()
            #     sum1 += v.sum()
            # for k, v in policy_model.denoiser.state_dict().items():
            #     cnt2 += v.numel()
            #     sum2 += v.sum()
            # # print(cnt1, cnt2, cnt1 - cnt2)
            # print(sum1.item(), sum2.item(), (sum1 - sum2).item())
            optimizer.zero_grad(set_to_none=True)
            del recons_motion, reward_input_motion
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
            # assert abs(prex.item() - x_i.sum().item()) < 1e-8
            x_i = x_i.detach().requires_grad_()  # 关键修复
            
        return x_i, reward