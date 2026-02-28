# t2m_runtime_clip_only.py
"""
CLIP-only 版本的 T2MRuntime，不加载 Qwen3 LLM，避免 tokenizer 问题。
使用 null ctxt features 替代 LLM 输出。
"""
import os
import threading
import time
import uuid
from typing import List, Optional, Tuple, Union

import torch
import yaml

from ..prompt_engineering.prompt_rewrite import PromptRewriter
from .loaders import load_object
from .visualize_mesh_web import save_visualization_data, generate_static_html_content

try:
    import fbx

    FBX_AVAILABLE = True
    print(">>> FBX module found.")
except ImportError:
    FBX_AVAILABLE = False
    print(">>> FBX module not found.")


def _get_local_ip():
    import subprocess

    result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for ip in result.stdout.strip().split():
            if not ip.startswith("127.") and not ip.startswith("172.17."):
                return ip
    return "localhost"


def _now():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"


class T2MRuntimeClipOnly:
    """
    CLIP-only 版本的 T2MRuntime。
    不加载 Qwen3 LLM，使用 null ctxt features 替代。
    """
    def __init__(
        self,
        config_path: str,
        ckpt_name: str = "latest.ckpt",
        skip_text: bool = False,
        device_ids: Union[list[int], None] = None,
        skip_model_loading: bool = False,
        force_cpu: bool = False,
        disable_prompt_engineering: bool = True,
        prompt_engineering_host: Optional[str] = None,
        prompt_engineering_model_path: Optional[str] = None,
    ):
        self.config_path = config_path
        self.ckpt_name = ckpt_name
        self.skip_text = skip_text
        self.prompt_engineering_host = prompt_engineering_host
        self.prompt_engineering_model_path = prompt_engineering_model_path
        self.disable_prompt_engineering = True  # CLIP-only 模式强制禁用 prompt engineering
        self.skip_model_loading = skip_model_loading
        self.local_ip = _get_local_ip()

        if force_cpu:
            print(">>> [INFO] CPU mode enabled via HY_MOTION_DEVICE=cpu environment variable")
            self.device_ids = []
        elif torch.cuda.is_available():
            all_ids = list(range(torch.cuda.device_count()))
            self.device_ids = all_ids if device_ids is None else [i for i in device_ids if i in all_ids]
        else:
            self.device_ids = []

        self.pipelines = []
        self._gpu_load = []
        self._lock = threading.Lock()
        self._loaded = False

        self.prompt_rewriter = None  # CLIP-only 模式不使用 prompt rewriter
        
        if self.skip_model_loading:
            print(">>> [WARNING] Checkpoint not found, will use randomly initialized model weights")
        self.load()
        self.fbx_available = FBX_AVAILABLE
        if self.fbx_available:
            try:
                from .smplh2woodfbx import SMPLH2WoodFBX

                self.fbx_converter = SMPLH2WoodFBX()
            except Exception as e:
                print(f">>> Failed to initialize FBX converter: {e}")
                self.fbx_available = False
                self.fbx_converter = None
        else:
            self.fbx_converter = None
            print(">>> FBX module not found. FBX export will be disabled.")

        device_info = self.device_ids if self.device_ids else "cpu"
        print(f">>> T2MRuntimeClipOnly loaded in IP {self.local_ip}, devices={device_info}")
        print(">>> Using CLIP-only mode (Qwen3 LLM disabled)")

    def load(self):
        if self._loaded:
            return
        print(f">>> Loading model from {self.config_path}...")

        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # 修改 text encoder 配置，禁用 LLM
        if "network_module_args" in config:
            if "text_encoder_module_args" in config["network_module_args"]:
                # 设置 llm_type 为 None 以禁用 Qwen3
                config["network_module_args"]["text_encoder_module_args"]["llm_type"] = None
                print(">>> Disabled LLM (Qwen3) in text encoder config")

        allow_empty_ckpt = self.skip_model_loading

        if not self.device_ids:
            pipeline = load_object(
                config["train_pipeline"],
                config["train_pipeline_args"],
                network_module=config["network_module"],
                network_module_args=config["network_module_args"],
            )
            device = torch.device("cpu")
            pipeline.load_in_demo(
                self.ckpt_name,
                build_text_encoder=not self.skip_text,
                allow_empty_ckpt=allow_empty_ckpt,
            )
            pipeline.to(device)
            self.pipelines = [pipeline]
            self._gpu_load = [0]
        else:
            for gid in self.device_ids:
                p = load_object(
                    config["train_pipeline"],
                    config["train_pipeline_args"],
                    network_module=config["network_module"],
                    network_module_args=config["network_module_args"],
                )
                p.load_in_demo(
                    self.ckpt_name,
                    build_text_encoder=not self.skip_text,
                    allow_empty_ckpt=allow_empty_ckpt,
                )
                p.to(torch.device(f"cuda:{gid}"))
                self.pipelines.append(p)
            self._gpu_load = [0] * len(self.pipelines)

        self._loaded = True

    def _acquire_pipeline(self) -> int:
        while True:
            with self._lock:
                for i in range(len(self._gpu_load)):
                    if self._gpu_load[i] == 0:
                        self._gpu_load[i] = 1
                        return i
            time.sleep(0.01)

    def _release_pipeline(self, idx: int):
        with self._lock:
            self._gpu_load[idx] = 0

    def generate_motion(
        self,
        text: str,
        seeds_csv: str,
        duration: float,
        cfg_scale: float,
        output_format: str = "fbx",
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        original_text: Optional[str] = None,
        use_special_game_feat: bool = False,
    ) -> Tuple[Union[str, list[str]], dict]:
        self.load()
        seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip() != ""]
        pi = self._acquire_pipeline()
        try:
            pipeline = self.pipelines[pi]
            pipeline.eval()
            device = next(pipeline.parameters()).device
            batch_size = len(seeds) if seeds else 1

            # CLIP-only 模式：使用 CLIP 编码文本，使用 null ctxt 替代 LLM
            if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                # 只使用 CLIP 编码
                vtxt_raw = pipeline.text_encoder.encode_sentence_emb([text] * batch_size)
                
                # 使用 null ctxt features 替代 LLM 输出
                hidden_state_dict = {
                    "text_vec_raw": vtxt_raw.to(device),
                    "text_ctxt_raw": pipeline.null_ctxt_input.expand(batch_size, -1, -1).to(device),
                    "text_ctxt_raw_length": torch.tensor([1] * batch_size, device=device),
                }
                
                model_output = pipeline.generate(
                    text,
                    seeds,
                    duration,
                    cfg_scale=cfg_scale,
                    use_special_game_feat=use_special_game_feat,
                    hidden_state_dict=hidden_state_dict,
                )
            else:
                # 如果没有 text encoder，使用完全的 null features
                hidden_state_dict = {
                    "text_vec_raw": pipeline.null_vtxt_feat.expand(batch_size, -1, -1).to(device),
                    "text_ctxt_raw": pipeline.null_ctxt_input.expand(batch_size, -1, -1).to(device),
                    "text_ctxt_raw_length": torch.tensor([1] * batch_size, device=device),
                }
                model_output = pipeline.generate(
                    text,
                    seeds,
                    duration,
                    cfg_scale=1.0,  # 无条件生成时禁用 CFG
                    use_special_game_feat=False,
                    hidden_state_dict=hidden_state_dict,
                )
        finally:
            self._release_pipeline(pi)

        ts = _now()
        save_data, base_filename = save_visualization_data(
            output=model_output,
            text=text if original_text is None else original_text,
            rewritten_text=text,
            timestamp=ts,
            output_dir=output_dir,
            output_filename=output_filename,
        )

        html_content = self._generate_html_content(
            timestamp=ts,
            file_path=base_filename,
            output_dir=output_dir,
        )

        if output_format == "fbx" and not self.fbx_available:
            print(">>> Warning: FBX export requested but FBX SDK is not available. Falling back to dict format.")
            output_format = "dict"

        if output_format == "fbx" and self.fbx_available:
            fbx_files = self._generate_fbx_files(
                visualization_data=save_data,
                output_dir=output_dir,
                fbx_filename=output_filename,
            )
            return html_content, fbx_files, model_output
        elif output_format == "dict":
            return html_content, [], model_output
        else:
            raise ValueError(f">>> Invalid output format: {output_format}")

    def _generate_html_content(
        self,
        timestamp: str,
        file_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        print(f">>> Generating static HTML content, timestamp: {timestamp}")
        gradio_dir = output_dir if output_dir is not None else "output/gradio"

        try:
            html_content = generate_static_html_content(
                folder_name=gradio_dir,
                file_name=file_path,
                hide_captions=False,
            )

            print(f">>> Static HTML content generated for: {file_path}")
            return html_content

        except Exception as e:
            print(f">>> Failed to generate static HTML content: {e}")
            import traceback

            traceback.print_exc()
            return f"<html><body><h1>Error generating visualization</h1><p>{str(e)}</p></body></html>"

    def _generate_fbx_files(
        self,
        visualization_data: dict,
        output_dir: Optional[str] = None,
        fbx_filename: Optional[str] = None,
    ) -> List[str]:
        assert "smpl_data" in visualization_data, "smpl_data not found in visualization_data"
        fbx_files = []
        if output_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(root_dir, "output", "gradio")

        smpl_data_list = visualization_data["smpl_data"]

        unique_id = str(uuid.uuid4())[:8]
        text = visualization_data["text"]
        timestamp = visualization_data["timestamp"]
        for bb in range(len(smpl_data_list)):
            smpl_data = smpl_data_list[bb]
            if fbx_filename is None:
                fbx_filename_bb = f"{timestamp}_{unique_id}_{bb:03d}.fbx"
            else:
                fbx_filename_bb = f"{fbx_filename}_{bb:03d}.fbx"
            fbx_path = os.path.join(output_dir, fbx_filename_bb)
            success = self.fbx_converter.convert_npz_to_fbx(smpl_data, fbx_path)
            if success:
                fbx_files.append(fbx_path)
                print(f"\t>>> FBX file generated: {fbx_path}")
                txt_path = fbx_path.replace(".fbx", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                fbx_files.append(txt_path)

        return fbx_files
