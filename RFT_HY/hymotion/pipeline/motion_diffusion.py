import os
import os.path as osp
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import Tensor
from torchdiffeq import odeint

from ..utils.geometry import (
    matrix_to_quaternion,
    quaternion_fix_continuity,
    quaternion_to_matrix,
    rot6d_to_rotation_matrix,
    rotation_matrix_to_rot6d,
)
from ..utils.loaders import load_object
from ..utils.motion_process import smooth_rotation
from ..utils.type_converter import get_module_device
from .body_model import WoodenMesh


def length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """
        lengths: (B, 1)
        max_len: int
    Returns: (B, max_len)
    """
    assert lengths.max() <= max_len, f"lengths.max()={lengths.max()} > max_len={max_len}"
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths
    return mask


def start_end_frame_to_mask(start_frame: Tensor, end_frame: Tensor, max_len: int) -> Tensor:
    assert (start_frame >= 0).all() and (end_frame >= 0).all(), f"start_frame={start_frame}, end_frame={end_frame}"
    lengths = end_frame - start_frame + 1
    assert lengths.max() <= max_len, f"lengths.max()={lengths.max()} > max_len={max_len}"
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    batch_size = start_frame.shape[0]
    arange_ids = torch.arange(max_len, device=start_frame.device).unsqueeze(0).expand(batch_size, max_len)
    mask = (arange_ids >= start_frame.unsqueeze(1)) & (arange_ids <= end_frame.unsqueeze(1))
    return mask


def randn_tensor(
    shape,
    generator=None,
    device=None,
    dtype=None,
    layout=None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`.

    When passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the
    tensor is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


class MotionGeneration(torch.nn.Module):
    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        text_encoder_module: str,
        text_encoder_cfg: dict,
        mean_std_dir: str,
        motion_type="auto",
        **kwargs,
    ):
        super().__init__()
        # build models and parameters
        self._network_module_args = deepcopy(network_module_args)
        self.motion_transformer = load_object(network_module, network_module_args)
        self._text_encoder_module = text_encoder_module
        self._text_encoder_cfg = deepcopy(text_encoder_cfg)
        self.motion_type = motion_type

        self.null_vtxt_feat = torch.nn.Parameter(
            torch.randn(1, 1, self._network_module_args.get("vtxt_input_dim", 768))
        )
        self.null_ctxt_input = torch.nn.Parameter(
            torch.randn(1, 1, self._network_module_args.get("ctxt_input_dim", 4096))
        )
        self.special_game_vtxt_feat = torch.nn.Parameter(
            torch.randn(1, 1, self._network_module_args.get("vtxt_input_dim", 768))
        )
        self.special_game_ctxt_feat = torch.nn.Parameter(
            torch.randn(1, 1, self._network_module_args.get("ctxt_input_dim", 4096))
        )
        # build buffer
        self.mean_std_dir = mean_std_dir
        self._parse_buffer(self.motion_type)

        self.output_mesh_fps = kwargs.get("output_mesh_fps", 30)
        self.train_frames = kwargs.get("train_frames", 360)
        self.uncondition_mode = kwargs.get("uncondition_mode", False)
        self.enable_ctxt_null_feat = kwargs.get("enable_ctxt_null_feat", False)
        self.enable_special_game_feat = kwargs.get("enable_special_game_feat", False)
        self.random_generator_on_gpu = kwargs.get("random_generator_on_gpu", True)

    def _parse_buffer(self, mode: str) -> None:
        self.body_model = WoodenMesh()
        self._find_motion_type(mode=mode)
        self._load_mean_std()

    def _load_mean_std(self, mean_std_name: Optional[str] = None) -> None:
        mean_std_name = self.mean_std_dir if mean_std_name is None else mean_std_name
        if mean_std_name is not None and osp.isdir(mean_std_name):
            mean = torch.from_numpy(np.load(osp.join(mean_std_name, "Mean.npy"))).float()
            std = torch.from_numpy(np.load(osp.join(mean_std_name, "Std.npy"))).float()
            self._assert_motion_dimension(mean.unsqueeze(0), std.unsqueeze(0))
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        else:
            print(
                f"[{self.__class__.__name__}] No mean_std found, using blank mean_std, "
                f"self.mean_std_dir={self.mean_std_dir}"
            )
            self.register_buffer("mean", torch.zeros(1))
            self.register_buffer("std", torch.ones(1))

    def _assert_motion_dimension(self, mean: Tensor, std: Tensor) -> None:
        assert mean.shape == std.shape, f"mean.shape={mean.shape} != std.shape={std.shape}"
        assert mean.ndim == 2, f"mean.ndim={mean.ndim} != 2"
        assert mean.shape == (1, 201), f"mean.shape={mean.shape} != (1, 201)"

    def _find_motion_type(self, mode: str) -> None:
        if mode == "auto":
            self.motion_type = "o6dp"
        else:
            self.motion_type = mode

    def set_epoch(self, epoch) -> None:
        self.current_epoch = epoch

    def load_in_demo(
        self,
        ckpt_name: str,
        build_text_encoder: bool = True,
        allow_empty_ckpt: bool = False,
    ) -> None:
        if not allow_empty_ckpt:
            if not os.path.exists(ckpt_name):
                import warnings

                warnings.warn(f"Checkpoint {ckpt_name} not found, skipping model loading")
            else:
                checkpoint = torch.load(ckpt_name, map_location="cpu", weights_only=False)
                self.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.motion_transformer.eval()
        if build_text_encoder and not self.uncondition_mode:
            self.text_encoder = load_object(self._text_encoder_module, self._text_encoder_cfg)
            self.text_encoder.to(get_module_device(self))

    @torch.no_grad()
    def encode_text(self, text: Dict[str, List[str]]) -> Dict[str, Tensor]:
        if not hasattr(self, "text_encoder"):
            self.text_encoder = load_object(self._text_encoder_module, self._text_encoder_cfg)
            self.text_encoder.to(get_module_device(self))
        text = text["text"]
        vtxt_input, ctxt_input, ctxt_length = self.text_encoder.encode(text=text)
        return {
            "text_vec_raw": vtxt_input,
            "text_ctxt_raw": ctxt_input,
            "text_ctxt_raw_length": ctxt_length,
        }

    def decode_motion_from_latent(self, latent: Tensor, should_apply_smooothing: bool = True) -> Dict[str, Tensor]:
        std_zero = self.std < 1e-3
        std = torch.where(std_zero, torch.zeros_like(self.std), self.std)
        latent_denorm = latent * std + self.mean
        return self._decode_o6dp(
            latent_denorm,
            num_joints=22,
            rel_trans=False,
            should_apply_smooothing=should_apply_smooothing,
        )

    def _decode_o6dp(
        self,
        latent_denorm: torch.Tensor,
        num_joints: int,
        rel_trans: bool = False,
        should_apply_smooothing: bool = True,
    ) -> dict:
        device = get_module_device(self)
        B, L = latent_denorm.shape[:2]
        nj = num_joints
        body_n = nj - 1

        if not rel_trans:
            transl = latent_denorm[..., 0:3].clone()
        else:
            transl = torch.cumsum(latent_denorm[..., 0:3].clone(), dim=1) / self.output_mesh_fps
        root_rot6d = latent_denorm[..., 3:9].reshape(B, L, 1, 6).clone()

        body6d_start = 9
        body6d_end = body6d_start + body_n * 6
        body_rot6d_full = latent_denorm[..., body6d_start:body6d_end].clone().reshape(B, L, body_n, 6)

        # 52 joints need to be split into hands
        left_hand_pose = right_hand_pose = None
        if nj == 52:
            body_rot6d = body_rot6d_full[:, :, :21, :].clone()
            left_hand_pose = body_rot6d_full[:, :, 21:36, :].clone()
            right_hand_pose = body_rot6d_full[:, :, 36:51, :].clone()
        else:
            body_rot6d = body_rot6d_full

        if left_hand_pose is not None and right_hand_pose is not None:
            body_full = torch.cat([body_rot6d, left_hand_pose, right_hand_pose], dim=2)
        else:
            body_full = body_rot6d
        rot6d = torch.cat([root_rot6d, body_full], dim=2)  # (B, L, nj, 6)
        if should_apply_smooothing:
            # only apply slerp smoothing to the first 22 joints (non-finger joints)
            rot6d_body = rot6d[:, :, :22, :]  # (B, L, 22, 6)
            rot6d_fingers = rot6d[:, :, 22:, :]  # (B, L, J-22, 6)
            rot6d_body_smooth = self.smooth_with_slerp(rot6d_body, sigma=1.0)
            rot6d_smooth = torch.cat([rot6d_body_smooth, rot6d_fingers], dim=2)
        else:
            rot6d_smooth = rot6d
        root_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 0, :])  # (B, L, 3, 3)

        transl_fixed = transl.detach()
        if should_apply_smooothing:
            transl_smooth = self.smooth_with_savgol(transl_fixed.detach(), window_length=11, polyorder=5)
        else:
            transl_smooth = transl_fixed

        if self.body_model is not None:
            print(
                f"{self.__class__.__name__} rot6d_smooth shape: {rot6d_smooth.shape}, transl_smooth shape: {transl_smooth.shape}"
            )
            with torch.no_grad():
                vertices_all = []
                k3d_all = []
                for bs in range(rot6d_smooth.shape[0]):
                    out = self.body_model.forward({"rot6d": rot6d_smooth[bs], "trans": transl_smooth[bs]})
                    vertices_all.append(out["vertices"])
                    k3d_all.append(out["keypoints3d"])
                vertices = torch.stack(vertices_all, dim=0)
                k3d = torch.stack(k3d_all, dim=0)
            print(f"{self.__class__.__name__} vertices shape: {vertices.shape}, k3d shape: {k3d.shape}")
            # align with the ground
            min_y = vertices[..., 1].amin(dim=(1, 2), keepdim=True)  # (B, 1, 1)
            print(f"{self.__class__.__name__} min_y: {min_y}")
            k3d = k3d.clone()
            k3d[..., 1] -= min_y  # (B, L, J) - (B, 1, 1)
            transl_smooth = transl_smooth.clone()
            transl_smooth[..., 1] -= min_y.squeeze(-1).to(device)  # (B, L) - (B, 1)
        else:
            k3d = torch.zeros(B, L, nj, 3, device=device)

        return dict(
            latent_denorm=latent_denorm.cpu().detach(),  # (B, L, 201)
            keypoints3d=k3d.cpu().detach(),  # (B, L, J, 3)
            rot6d=rot6d_smooth.cpu().detach(),  # (B, L, J, 6)
            transl=transl_smooth.cpu().detach(),  # (B, L, 3)
            root_rotations_mat=root_rotmat_smooth.cpu().detach(),  # (B, L, 3, 3)
        )

    @staticmethod
    def smooth_with_savgol(input: torch.Tensor, window_length: int = 9, polyorder: int = 5) -> torch.Tensor:
        if len(input.shape) == 2:
            is_batch = False
            input = input.unsqueeze(0)
        else:
            is_batch = True
        input_np = input.cpu().numpy()
        input_smooth_np = np.empty_like(input_np, dtype=np.float32)
        for b in range(input_np.shape[0]):
            for j in range(input_np.shape[2]):
                input_smooth_np[b, :, j] = savgol_filter(input_np[b, :, j], window_length, polyorder)
        input_smooth = torch.from_numpy(input_smooth_np).to(input)
        if not is_batch:
            input_smooth = input_smooth.squeeze(0)
        return input_smooth

    @staticmethod
    def smooth_with_slerp(input: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        def fix_time_continuity(q: Tensor, time_dim: int = -3):
            shape = q.shape
            qv = q.moveaxis(time_dim, 0).contiguous().view(shape[time_dim], -1, 4)
            qv = quaternion_fix_continuity(qv)
            return qv.view(shape[time_dim], *shape[:time_dim], *shape[time_dim + 1 :]).moveaxis(0, time_dim)

        num_joints = input.shape[2]
        RR = rot6d_to_rotation_matrix(input)
        qq = matrix_to_quaternion(RR)
        qq_np = fix_time_continuity(qq, time_dim=1).cpu().numpy()
        qq_s_np = smooth_rotation(
            qq_np,
            sigma=sigma,
        )
        input_smooth = rotation_matrix_to_rot6d(quaternion_to_matrix(torch.from_numpy(qq_s_np)))
        return input_smooth.to(input.device)

    @staticmethod
    def noise_from_seeds(
        latent: Tensor, seeds: Union[int, List[int]], seed_start: int = 0, random_generator_on_gpu: bool = True
    ) -> Tensor:
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        noise_list = []
        B = latent.shape[0]
        shape = (B, *latent.shape[1:])
        for seed in seeds:
            if random_generator_on_gpu:
                generator = torch.Generator(device=latent.device).manual_seed(seed + seed_start)
                noise_sample = randn_tensor(shape, generator=generator, device=latent.device, dtype=latent.dtype)
            else:
                generator = torch.Generator().manual_seed(seed + seed_start)
                noise_sample = randn_tensor(shape, generator=generator, dtype=latent.dtype).to(latent.device)
            noise_list.append(noise_sample)
        return torch.cat(noise_list, dim=0)

    def _maybe_inject_source_token(
        self,
        vtxt_input: Tensor,
        ctxt_input: Tensor,
        ctxt_mask_temporal: Tensor,
        sources: Optional[List[str]],
        trigger_sources: Optional[set] = None,
        prob: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if (sources is None or trigger_sources is None) or not self.enable_special_game_feat:
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        B, Lc, Dc = ctxt_input.shape
        assert (
            isinstance(sources, (list, tuple)) and len(sources) == B
        ), f"sources length should be equal to batch: {len(sources)} vs {B}"

        trig = set(s.lower() for s in trigger_sources)
        src_mask = torch.tensor(
            [str(s).lower() in trig for s in sources], dtype=torch.bool, device=ctxt_input.device
        )  # (B,)
        if not src_mask.any():
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        rand_mask = (
            torch.rand(B, device=ctxt_input.device) < prob
            if self.training
            else torch.BoolTensor(B).fill_(True).to(ctxt_input.device)
        )
        apply_mask = src_mask & rand_mask
        if not apply_mask.any():
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        # vtxt: only add mixture to the hit samples
        vtxt_token = self.special_game_vtxt_feat.to(vtxt_input).expand(B, 1, -1)
        vtxt_input = vtxt_input + vtxt_token * apply_mask.view(B, 1, 1).to(vtxt_input.dtype)

        # calculate the current effective length of each sample
        if ctxt_mask_temporal.dtype == torch.bool:
            cur_len = ctxt_mask_temporal.sum(dim=1).long()  # (B,)
        else:
            cur_len = (ctxt_mask_temporal > 0).sum(dim=1).long()

        # for the "not full" hit samples,
        # write the special token at the cur_len position,
        # and set the mask to True
        can_inplace = apply_mask & (cur_len < Lc)
        b_inplace = torch.nonzero(can_inplace, as_tuple=False).squeeze(1)  # (K,)
        if b_inplace.numel() > 0:
            pos = cur_len[b_inplace]  # (K,)
            token = self.special_game_ctxt_feat.squeeze(0).squeeze(0).to(ctxt_input)  # (Dc,)
            ctxt_input[b_inplace, pos, :] = token.unsqueeze(0).expand(b_inplace.numel(), Dc)
            if ctxt_mask_temporal.dtype == torch.bool:
                ctxt_mask_temporal[b_inplace, pos] = True
            else:
                ctxt_mask_temporal[b_inplace, pos] = 1

        # if there are "full" hit samples, need to pad one:
        # the full samples write the special token at the new position,
        # other samples pad zero and mask=False
        need_expand = (apply_mask & (cur_len >= Lc)).any()
        if need_expand:
            suffix = torch.zeros((B, 1, Dc), dtype=ctxt_input.dtype, device=ctxt_input.device)
            full_hit = apply_mask & (cur_len >= Lc)
            b_full = torch.nonzero(full_hit, as_tuple=False).squeeze(1)
            if b_full.numel() > 0:
                suffix[b_full, 0, :] = (
                    self.special_game_ctxt_feat.expand(b_full.numel(), 1, -1).to(ctxt_input).squeeze(1)
                )
            ctxt_input = torch.cat([ctxt_input, suffix], dim=1)

            if ctxt_mask_temporal.dtype == torch.bool:
                suffix_mask = torch.zeros((B, 1), dtype=torch.bool, device=ctxt_input.device)
                suffix_mask[b_full, 0] = True
            else:
                suffix_mask = torch.zeros((B, 1), dtype=ctxt_mask_temporal.dtype, device=ctxt_input.device)
                suffix_mask[b_full, 0] = 1
            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal, suffix_mask], dim=1)

        return vtxt_input, ctxt_input, ctxt_mask_temporal


class MotionFlowMatching(MotionGeneration):
    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        text_encoder_module: str,
        text_encoder_cfg: dict,
        noise_scheduler_cfg: dict = {"method": "euler"},
        infer_noise_scheduler_cfg: dict = {"validation_steps": 50},
        mean_std_dir: Optional[str] = None,
        losses_cfg: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            network_module=network_module,
            network_module_args=network_module_args,
            text_encoder_module=text_encoder_module,
            text_encoder_cfg=text_encoder_cfg,
            losses_cfg=losses_cfg,
            mean_std_dir=(mean_std_dir if mean_std_dir is not None else test_cfg.get("mean_std_dir", None)),
            **kwargs,
        )
        # build scheduler
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg)
        self._infer_noise_scheduler_cfg = deepcopy(infer_noise_scheduler_cfg)
        # additional cfg
        self.train_cfg = deepcopy(train_cfg) if train_cfg else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg else dict()
        self._parse_test_cfg()

    def _parse_test_cfg(self) -> None:
        self.validation_steps = self._infer_noise_scheduler_cfg["validation_steps"]
        self.text_guidance_scale = self.test_cfg.get("text_guidance_scale", 1)

    @torch.no_grad()
    def generate(
        self,
        text: Union[str, List[str]],
        seed_input: List[int],
        duration_slider: int,
        cfg_scale: Optional[float] = None,
        use_special_game_feat: bool = False,
        hidden_state_dict=None,
        length=None,
    ) -> Dict[str, Any]:
        device = get_module_device(self)
        if length is None:
            length = int(round(duration_slider * self.output_mesh_fps))
        assert (
            0 < length < 5000
        ), f"input duration_slider must be in (0, {5000/self.output_mesh_fps}] due to rope, but got {duration_slider}"
        if length > self.train_frames or length < min(self.train_frames, 20):
            print(f">>> given length is too long or too short, got {length}, will be truncated")
            length = min(length, self.train_frames)
            length = max(length, min(self.train_frames, 20))

        repeat = len(seed_input)
        if isinstance(text, list):
            assert len(text) == repeat, f"len(text) must equal len(seed_input), got {len(text)} vs {repeat}"
            text_list = text
        elif isinstance(text, str):
            text_list = [text] * repeat
        else:
            raise TypeError(f"Unsupported text type: {type(text)}")

        if not self.uncondition_mode:
            if hidden_state_dict is None:
                hidden_state_dict = self.encode_text({"text": text_list})
            vtxt_input = hidden_state_dict["text_vec_raw"]
            ctxt_input = hidden_state_dict["text_ctxt_raw"]
            ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
            # check shape
            if len(vtxt_input.shape) == 2 and len(ctxt_input.shape) == 2:
                vtxt_input = vtxt_input[None].repeat(repeat, 1, 1)
                ctxt_input = ctxt_input[None].repeat(repeat, 1, 1)
                ctxt_length = ctxt_length.repeat(repeat)
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            sources = None if not use_special_game_feat else ["Game"] * repeat
            vtxt_input, ctxt_input, ctxt_mask_temporal = self._maybe_inject_source_token(
                vtxt_input, ctxt_input, ctxt_mask_temporal, sources, trigger_sources={"Taobao", "Game"}
            )
        else:
            vtxt_input = self.null_vtxt_feat.expand(repeat, 1, -1)
            ctxt_input = self.null_ctxt_input.expand(repeat, 1, -1)
            ctxt_length = torch.tensor([1]).expand(repeat)
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1]).expand(repeat, -1)
        assert len(vtxt_input.shape) == 3, f"vtxt_input.shape: {vtxt_input.shape}, should be (B, 1, D)"
        assert len(ctxt_input.shape) == 3, f"ctxt_input.shape: {ctxt_input.shape}, should be (B, 1, D)"
        assert len(ctxt_length.shape) == 1, f"ctxt_length.shape: {ctxt_length.shape}, should be (B,)"

        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
        x_length = torch.LongTensor([length] * repeat).to(device)
        x_mask_temporal = length_to_mask(x_length, self.train_frames)

        text_guidance_scale = cfg_scale if cfg_scale is not None else self.text_guidance_scale
        do_classifier_free_guidance = text_guidance_scale > 1.0 and not self.uncondition_mode
        if do_classifier_free_guidance is True:
            silent_text_feat = self.null_vtxt_feat.expand(*vtxt_input.shape)
            vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)

            if self.enable_ctxt_null_feat:
                silent_ctxt_input = self.null_ctxt_input.expand(*ctxt_input.shape)
            else:
                silent_ctxt_input = ctxt_input
            ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)

            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

        def fn(t: Tensor, x: Tensor) -> Tensor:
            # predict flow
            x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
            x_pred = self.motion_transformer(
                x=x_input,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )
            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + text_guidance_scale * (x_pred_text - x_pred_basic)
            return x_pred

        # duplicate test corner for inner time step oberservation
        t = torch.linspace(0, 1, self.validation_steps + 1, device=device)
        y0 = self.noise_from_seeds(
            torch.zeros(
                1,
                self.train_frames,
                self._network_module_args["input_dim"],
                device=device,
            ),
            seed_input,
            random_generator_on_gpu=self.random_generator_on_gpu,
        )
        with torch.no_grad():
            trajectory = odeint(fn, y0, t, **self._noise_scheduler_cfg)
        sampled = trajectory[-1][:, :length, ...].clone()
        assert isinstance(sampled, Tensor), f"sampled must be a Tensor, but got {type(sampled)}"

        output_dict = self.decode_motion_from_latent(sampled, should_apply_smooothing=True)

        return {
            **output_dict,
            "text": text,
        }


if __name__ == "__main__":
    # python -m hymotion.pipeline.motion_diffusion
    import time

    import torch

    device = "cuda:0"
    bsz, input_dim = 64, 272
    seq_lens = [90, 180, 360]
    ctxt_seq_lens = 64
    warmup = 5
    repeats = 100

    network_module = "hymotion/network/hymotion_mmdit.HunyuanMotionMMDiT"
    network_module_args = {
        "input_dim": input_dim,
        "feat_dim": 512,
        "ctxt_input_dim": 4096,
        "vtxt_input_dim": 768,
        "num_layers": 12,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "mask_mode": "narrowband",
    }
    text_encoder_module = "hymotion/network/text_encoders/text_encoder.HYTextModel"
    text_encoder_cfg = {"llm_type": "qwen3", "max_length_llm": ctxt_seq_lens}

    # ================================ FM_MMDiT ================================
    FM_MMDiT = MotionFlowMatching(
        network_module=network_module,
        network_module_args=network_module_args,
        text_encoder_module=text_encoder_module,
        text_encoder_cfg=text_encoder_cfg,
        noise_scheduler_module={"method": "euler"},
        infer_noise_scheduler_cfg={"validation_steps": 50},
        train_cfg={"cond_mask_prob": 0.1},
        test_cfg={
            "text_guidance_scale": 1.5,
        },
    ).to(device)
