import argparse
import copy
import json
import logging
import os
import multiprocessing as mp
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from utils import (
    DatasetSubmissionConfig,
    basename_key,
    build_candidate_view,
    candidate_video_path,
    ensure_ext,
    infer_key_to_submission_id_from_video_dir,
    load_embedding_store,
    load_manifest_map,
    rerank_predictions,
    source_key_aliases,
    source_video_path,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

logger = logging.getLogger(__name__)


DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Retrieve the target video that matches the reference video after applying the edit instruction. "
    "The correct target should satisfy the requested visual change while preserving unchanged visual context, "
    "including objects, actions, state changes, scene, camera framing, and temporal progression."
)

DEFAULT_DOCUMENT_INSTRUCTION = (
    "Represent this candidate target video for composed video retrieval. Focus on visible objects, actions, "
    "state changes, scene, camera framing, and temporal progression."
)

DEFAULT_THINKING_REASONING_MODEL = "Qwen/Qwen3-VL-32B-Thinking"
RERANKER_OVERLAY_DIRNAME = "_qwen3vl_reranker_cross_encoder_overlay"


def _worker_error_path(output_file: str) -> str:
    return f"{output_file}.error.txt"


def _clear_worker_temp_files(output_file: str) -> None:
    for path in (output_file, _worker_error_path(output_file)):
        if os.path.exists(path):
            os.remove(path)


def _write_worker_error(output_file: str) -> None:
    error_path = _worker_error_path(output_file)
    os.makedirs(os.path.dirname(error_path) or ".", exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())


def _read_worker_error(output_file: str, limit: int = 8000) -> str:
    error_path = _worker_error_path(output_file)
    if not os.path.exists(error_path):
        return ""
    with open(error_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if len(text) > limit:
        return "... truncated worker traceback ...\n" + text[-limit:]
    return text


def _is_too_small_video_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "must be larger than factor" in message
        and "height:" in message
        and "width:" in message
    )


@dataclass
class OfficialQwenConfig:
    val_json: str
    output_json: str
    artifact_dir: str = "./artifacts/official_qwen3vl"

    embedding_model_name: str = "Qwen/Qwen3-VL-Embedding-8B"
    reranker_model_name: str = "Qwen/Qwen3-VL-Reranker-8B"
    device: Optional[str] = None
    attn_implementation: Optional[str] = "sdpa"
    torch_dtype: str = "bfloat16"
    reranker_true_token_id: int = 9693
    reranker_false_token_id: int = 2152
    reranker_backend: str = "auto"
    reranker_device_map: Optional[str] = None
    parallelism: str = "single"
    gpu_ids: Optional[List[int]] = None

    top_k: int = 50
    rerank_top_k: int = 0
    embedding_batch_size: int = 1
    reranker_batch_size: int = 1
    reranker_predict_chunk_size: int = 20
    reranker_fps: Optional[float] = None
    reranker_max_frames: Optional[int] = 16
    reranker_max_pixels: Optional[int] = None
    normalize_embeddings: bool = True
    exclude_source: bool = True
    keep_modification_text: bool = False
    include_debug_artifacts: bool = False
    include_reasoning_trace: bool = False
    reasoning_for_retrieval: bool = False
    reasoning_for_rerank: bool = False
    reasoning_cache_json: Optional[str] = None

    fps: float = 1.0
    max_frames: Optional[int] = 128
    max_pixels: Optional[int] = None
    video_decode_retries: int = 3
    video_decode_retry_sleep: float = 1.5

    query_template: str = "Reference video after edit instruction: {modification_text}"
    query_templates: Optional[List[str]] = None
    reasoning_query_template: str = (
        "Reference video after edit instruction: {modification_text}\n"
        "Reasoned target video description: {reasoning}"
    )
    reasoning_reranker_template: str = (
        "{modification_text}\n\n"
        "Reasoned target video description:\n{reasoning}"
    )
    query_fusion: str = "max"
    reasoning_retrieval_mode: str = "gated_residual"
    reasoning_min_query_similarity: float = 0.0
    reasoning_fusion_weight: float = 0.08
    reasoning_candidate_pool_k: int = 0
    reasoning_rrf_k: float = 60.0
    embedding_instruction: str = DEFAULT_RETRIEVAL_INSTRUCTION
    document_instruction: str = DEFAULT_DOCUMENT_INSTRUCTION
    reranker_instruction: str = DEFAULT_RETRIEVAL_INSTRUCTION

    fusion_embedding_weight: float = 0.4
    fusion_reranker_weight: float = 0.6
    submission_top_k: int = 50

    reasoning_model_name: str = DEFAULT_THINKING_REASONING_MODEL
    reasoning_device_map: Optional[str] = None
    reasoning_enable_thinking: Optional[bool] = True
    reasoning_decode_skip_special_tokens: bool = False
    reasoning_trace_part: str = "final"
    reasoning_tokens: int = 4096
    reasoning_prompt_style: str = "benchmark"
    reasoning_tag_mode: str = "none"
    reasoning_do_sample: bool = False
    reasoning_temperature: float = 0.2
    reasoning_top_p: float = 0.9
    reasoning_top_k: int = 50

    datasets: Dict[str, DatasetSubmissionConfig] = field(default_factory=dict)


def _require_sentence_transformers():
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "This official Qwen3-VL pipeline requires sentence-transformers. "
            "Install it with: pip install 'sentence-transformers>=5.4.0'"
        ) from exc
    return SentenceTransformer, CrossEncoder


def load_official_config(path: str) -> OfficialQwenConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    datasets = {}
    for name, raw in (data.get("datasets") or {}).items():
        datasets[name] = DatasetSubmissionConfig(
            video_dir=raw["video_dir"],
            gallery_npz=raw["gallery_npz"],
            video_extension=ensure_ext(raw.get("video_extension", ".mp4")),
            manifest_path=raw.get("manifest_path"),
            key_prefix=raw.get("key_prefix", ""),
            output_source_as_string=bool(raw.get("output_source_as_string", False)),
        )

    cfg = OfficialQwenConfig(
        val_json=data["val_json"],
        output_json=data["output_json"],
        datasets=datasets,
    )
    for key, value in data.items():
        if key == "datasets":
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _torch_dtype_value(name: str):
    import torch

    name = str(name or "").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return None


def _safe_symlink(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except FileExistsError:
        return


def _cross_encoder_logit_score_config_paths(model_dir: str) -> List[str]:
    modules_path = os.path.join(model_dir, "modules.json")
    paths: List[str] = []
    if os.path.exists(modules_path):
        try:
            with open(modules_path, "r", encoding="utf-8") as f:
                modules = json.load(f)
        except Exception as exc:
            logger.warning("Could not inspect reranker modules.json at %s: %s", modules_path, exc)
            modules = []
        for module in modules:
            module_type = str(module.get("type", ""))
            module_path = str(module.get("path", ""))
            if "LogitScore" in module_type and module_path:
                paths.append(os.path.join(module_path, "config.json"))
    if not paths and os.path.isdir(os.path.join(model_dir, "1_LogitScore")):
        paths.append(os.path.join("1_LogitScore", "config.json"))
    return paths


def _prepare_reranker_cross_encoder_overlay(cfg: OfficialQwenConfig) -> str:
    model_dir = os.path.abspath(os.path.expanduser(cfg.reranker_model_name))
    if not os.path.isdir(model_dir):
        return cfg.reranker_model_name

    logit_config_paths = _cross_encoder_logit_score_config_paths(model_dir)
    patch_paths = []
    for rel_path in logit_config_paths:
        src_path = os.path.join(model_dir, rel_path)
        needs_patch = not os.path.exists(src_path)
        if not needs_patch:
            try:
                with open(src_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                needs_patch = (
                    "true_token_id" not in existing
                    or "false_token_id" not in existing
                )
            except Exception:
                needs_patch = True
        if needs_patch:
            patch_paths.append(rel_path)
    if not patch_paths:
        return cfg.reranker_model_name

    overlay_dir = os.path.abspath(os.path.join(cfg.artifact_dir, RERANKER_OVERLAY_DIRNAME))
    os.makedirs(overlay_dir, exist_ok=True)
    patched_top_dirs = {rel_path.split(os.sep)[0] for rel_path in patch_paths}

    for name in os.listdir(model_dir):
        src = os.path.join(model_dir, name)
        dst = os.path.join(overlay_dir, name)
        if name in patched_top_dirs:
            if os.path.islink(dst):
                os.unlink(dst)
            os.makedirs(dst, exist_ok=True)
            if os.path.isdir(src):
                for child in os.listdir(src):
                    if child == "config.json":
                        continue
                    _safe_symlink(os.path.join(src, child), os.path.join(dst, child))
        else:
            _safe_symlink(src, dst)

    payload = {
        "true_token_id": int(cfg.reranker_true_token_id),
        "false_token_id": int(cfg.reranker_false_token_id),
    }
    for rel_path in patch_paths:
        src_path = os.path.join(model_dir, rel_path)
        if os.path.exists(src_path):
            try:
                with open(src_path, "r", encoding="utf-8") as f:
                    payload = {**json.load(f), **payload}
            except Exception:
                pass
        dst = os.path.join(overlay_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        temp_dst = f"{dst}.tmp.{os.getpid()}"
        with open(temp_dst, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(temp_dst, dst)

    logger.warning(
        "Created CrossEncoder metadata overlay at %s because local reranker metadata needs %s",
        overlay_dir,
        ", ".join(patch_paths),
    )
    return overlay_dir


def _auto_device(device: Optional[str]) -> Optional[str]:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return None
    return None


def load_embedding_model(cfg: OfficialQwenConfig):
    SentenceTransformer, _ = _require_sentence_transformers()
    model_kwargs = {}
    dtype = _torch_dtype_value(cfg.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    kwargs = {"model_kwargs": model_kwargs} if model_kwargs else {}
    device = _auto_device(cfg.device)
    if device:
        kwargs["device"] = device
    logger.info("Loading embedding model: %s", cfg.embedding_model_name)
    return SentenceTransformer(cfg.embedding_model_name, **kwargs)


def load_reranker_model(cfg: OfficialQwenConfig):
    _, CrossEncoder = _require_sentence_transformers()
    backend = str(cfg.reranker_backend).lower()
    if backend == "direct":
        return DirectQwen3VLReranker(cfg)
    if backend not in {"auto", "cross_encoder", "crossencoder", "ce"}:
        raise ValueError(f"Unsupported reranker_backend: {cfg.reranker_backend}")

    model_kwargs = {}
    dtype = _torch_dtype_value(cfg.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    kwargs = {"model_kwargs": model_kwargs} if model_kwargs else {}
    device = _auto_device(cfg.device)
    if device:
        kwargs["device"] = device
    reranker_model_name = _prepare_reranker_cross_encoder_overlay(cfg)
    logger.info("Loading reranker model: %s", reranker_model_name)
    try:
        model = CrossEncoder(reranker_model_name, **kwargs)
        logger.info("Loaded reranker backend: SentenceTransformers CrossEncoder")
        return model
    except Exception as exc:
        if backend in {"cross_encoder", "crossencoder", "ce"}:
            raise RuntimeError(
                "Requested reranker_backend=cross_encoder, but SentenceTransformers CrossEncoder "
                "could not load the Qwen3-VL reranker."
            ) from exc
        message = str(exc)
        is_missing_logit_config = "LogitScore.__init__()" in message and "true_token_id" in message
        is_wrong_auto_model = "Qwen3VLConfig" in message and "AutoModelForCausalLM" in message
        if not is_missing_logit_config and not is_wrong_auto_model:
            raise
        logger.warning(
            "CrossEncoder loading failed for Qwen3-VL reranker; falling back to direct "
            "Qwen3VLForConditionalGeneration scoring (true=%s, false=%s). Error: %s",
            cfg.reranker_true_token_id,
            cfg.reranker_false_token_id,
            exc,
        )
        return DirectQwen3VLReranker(cfg)


def _log_rerank_input_stats(submission: List[Dict[str, List[Dict[str, Any]]]], cfg: OfficialQwenConfig) -> None:
    candidate_counts = []
    total_pairs = 0
    for _, row in _flatten_sections(submission):
        n_candidates = len(row.get("video_target") or [])
        candidate_counts.append(n_candidates)
        total_pairs += min(max(cfg.rerank_top_k, 0), n_candidates)
    if not candidate_counts:
        logger.info("Rerank input stats: no rows found")
        return
    counts = np.asarray(candidate_counts, dtype=np.int32)
    logger.info(
        "Rerank input stats: rows=%d, rerank_top_k=%d, candidates min/median/max=%.0f/%.0f/%.0f, "
        "mean=%.1f, total model pairs=%d",
        len(candidate_counts),
        cfg.rerank_top_k,
        float(counts.min()),
        float(np.median(counts)),
        float(counts.max()),
        float(counts.mean()),
        total_pairs,
    )


def _clean_decoded_text(text: str) -> str:
    for token in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(token, "")
    return text.strip()


def _strip_think_block(text: str) -> str:
    cleaned = _clean_decoded_text(text)
    if "</think>" in cleaned.lower():
        return re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1].strip() or cleaned
    return re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip() or cleaned


def _extract_think_block(text: str) -> str:
    cleaned = _clean_decoded_text(text)
    match = re.search(r"<think>(.*?)</think>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    if "</think>" in cleaned.lower():
        return re.split(r"</think>", cleaned, flags=re.IGNORECASE)[0].strip()
    return cleaned


def _select_reasoning_trace_part(text: str, trace_part: str) -> str:
    mode = str(trace_part or "final").lower()
    if mode == "think":
        return _extract_think_block(text)
    if mode == "full":
        return _clean_decoded_text(text)
    return _strip_think_block(text)


def _apply_chat_template_with_thinking(processor, messages, enable_thinking: Optional[bool]):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def _reasoning_video_content(video_path: str, cfg: OfficialQwenConfig) -> Dict[str, Any]:
    content: Dict[str, Any] = {
        "type": "video",
        "video": f"file://{os.path.abspath(video_path)}",
        "fps": float(cfg.fps),
    }
    if cfg.max_frames is not None:
        content["max_frames"] = int(cfg.max_frames)
    if cfg.max_pixels is not None:
        content["max_pixels"] = int(cfg.max_pixels)
    return content


def _benchmark_reasoning_prompt(edit_instruction: str, tag_mode: str = "none") -> str:
    tag_instruction = (
        "Start the final paragraph with exactly one tag, <simple> or <detailed>, choosing <simple> only when the "
        "source and edit are visually minimal; otherwise choose <detailed>. "
        if str(tag_mode).lower() == "auto"
        else "Do not output <simple> or <detailed> tags. "
    )
    return (
        "Given the reference video and edit instruction below, reason about the edited target video.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "After the private thinking, write ONLY the final benchmark-aligned CoVR-R reasoning paragraph. "
        "The paragraph should be an external, polished explanation of how the edited target video follows from "
        "the source video and modification text. "
        f"{tag_instruction}"
        "Do not include greetings, self-talk, step lists, bullets, headings, markdown, or phrases such as "
        "\"Got it\", \"Okay\", \"Let's\", \"Wait\", \"Need to\", or \"I think\". "
        "Begin directly with the visual transformation, describing what object, subject, scene, background, "
        "lighting, camera/framing, or motion is replaced, removed, preserved, or newly emphasized. "
        "Connect the changes causally: explain how removing old elements shifts attention, how new objects or "
        "actions change the scene's focus, how timing or movement replaces the source action, and how background "
        "or lighting supports the target appearance. Write one cohesive paragraph with enough concrete visual "
        "detail for the edit. When the edit involves object, action, scene, state, camera, or motion changes, "
        "the paragraph should usually be around 160-220 words; for very simple edits, remain concise but still "
        "mention the preserved context and the visible target-side change."
    )


class QwenThinkingReasoner:
    """Thinking-model reasoning component used inside official submit generation."""

    def __init__(self, cfg: OfficialQwenConfig):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.cfg = cfg
        self.uses_sampling_kwargs = True
        self.torch = torch
        self.device = _auto_device(cfg.device) or "cpu"
        dtype = _torch_dtype_value(cfg.torch_dtype)
        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation
        if cfg.reasoning_device_map:
            model_kwargs["device_map"] = cfg.reasoning_device_map

        logger.info("Loading thinking reasoning model: %s", cfg.reasoning_model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.reasoning_model_name, **model_kwargs)
        if not cfg.reasoning_device_map and self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(cfg.reasoning_model_name)

    def generate(self, source_video: str, edit_instruction: str) -> str:
        from qwen_vl_utils import process_vision_info

        prompt = _benchmark_reasoning_prompt(edit_instruction, self.cfg.reasoning_tag_mode)
        messages = [
            {
                "role": "user",
                "content": [
                    _reasoning_video_content(source_video, self.cfg),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
        text = _apply_chat_template_with_thinking(self.processor, messages, self.cfg.reasoning_enable_thinking)
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="pt",
            padding=True,
        )
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": int(self.cfg.reasoning_tokens),
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": bool(self.cfg.reasoning_do_sample),
        }
        if self.cfg.reasoning_do_sample:
            gen_kwargs.update({
                "temperature": float(self.cfg.reasoning_temperature),
                "top_p": float(self.cfg.reasoning_top_p),
                "top_k": int(self.cfg.reasoning_top_k),
            })

        with self.torch.no_grad():
            use_autocast = (
                target_device.type == "cuda"
                and str(self.cfg.torch_dtype).lower() in {"bf16", "bfloat16", "fp16", "float16", "half"}
            )
            autocast_dtype = self.torch.bfloat16 if str(self.cfg.torch_dtype).lower() in {"bf16", "bfloat16"} else self.torch.float16
            if use_autocast:
                with self.torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    output_ids = self.model.generate(**inputs, **gen_kwargs)
            else:
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        trimmed = output_ids[:, input_length:]
        raw = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=bool(self.cfg.reasoning_decode_skip_special_tokens),
            clean_up_tokenization_spaces=True,
        )[0]
        return _select_reasoning_trace_part(raw, self.cfg.reasoning_trace_part)


def load_reasoning_model(cfg: OfficialQwenConfig):
    if cfg.reasoning_cache_json:
        return None
    if not (cfg.include_reasoning_trace or cfg.reasoning_for_retrieval):
        return None
    return QwenThinkingReasoner(cfg)


def _reasoning_cache_key(section_name: str, row: Dict[str, Any]) -> str:
    return f"{section_name}:{row.get('id')}"


def load_reasoning_cache(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items() if v is not None}
    cache = {}
    if isinstance(data, list):
        for section in data:
            if not isinstance(section, dict):
                continue
            for section_name, rows in section.items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if isinstance(row, dict) and row.get("reasoning_trace"):
                        trace = row["reasoning_trace"]
                        if isinstance(trace, list) and trace:
                            cache[_reasoning_cache_key(str(section_name), row)] = str(trace[0])
                        elif isinstance(trace, str):
                            cache[_reasoning_cache_key(str(section_name), row)] = trace
    return cache


def save_reasoning_cache(path: str, cache: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


class DirectQwen3VLReranker:
    """Direct Qwen3-VL reranker fallback for SentenceTransformers incompatibilities."""

    def __init__(self, cfg: OfficialQwenConfig):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.cfg = cfg
        self.torch = torch
        self.device = _auto_device(cfg.device) or "cpu"
        dtype = _torch_dtype_value(cfg.torch_dtype)
        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation
        if cfg.reranker_device_map:
            model_kwargs["device_map"] = cfg.reranker_device_map

        logger.info("Loading direct Qwen3-VL reranker model: %s", cfg.reranker_model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.reranker_model_name, **model_kwargs)
        if not cfg.reranker_device_map and self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(cfg.reranker_model_name)

    def predict(self, pairs, prompt: str, batch_size: int = 1, show_progress_bar: bool = False):
        scores = []
        iterator = tqdm(pairs, desc="Direct reranker", disable=not show_progress_bar)
        for query_item, doc_item in iterator:
            scores.append(self._score_pair(query_item, doc_item, prompt))
        return np.asarray(scores, dtype=np.float32)

    def _video_content(self, item: Dict[str, Any]) -> Dict[str, Any]:
        content = {
            "type": "video",
            "video": f"file://{os.path.abspath(item['video'])}",
            "fps": float(item.get("fps", self.cfg.fps)),
        }
        if item.get("max_frames") is not None:
            content["max_frames"] = int(item["max_frames"])
        if item.get("max_pixels") is not None:
            content["max_pixels"] = int(item["max_pixels"])
        return content

    def _score_pair(self, query_item: Dict[str, Any], doc_item: Dict[str, Any], prompt: str) -> float:
        from qwen_vl_utils import process_vision_info

        query_text = query_item.get("text", "")
        text = (
            f"{prompt}\n\n"
            "The first video is the reference/source video. The second video is the candidate target video.\n"
            f"Query edit instruction and retrieval intent:\n{query_text}\n\n"
            "Answer only true or false: is the second video a good target match for the edited first video?"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    self._video_content(query_item),
                    self._video_content(doc_item),
                    {"type": "text", "text": text},
                ],
            }
        ]
        images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[chat_text],
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="pt",
            padding=True,
        )
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self.torch.no_grad():
            with self.torch.autocast(
                device_type="cuda",
                dtype=self.torch.bfloat16,
                enabled=target_device.type == "cuda" and self.cfg.torch_dtype.lower() in {"bf16", "bfloat16"},
            ):
                outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]
        true_logit = logits[int(self.cfg.reranker_true_token_id)]
        false_logit = logits[int(self.cfg.reranker_false_token_id)]
        return float((true_logit - false_logit).detach().float().cpu().item())


def _media_item(
    video_path: str,
    cfg: OfficialQwenConfig,
    text: Optional[str] = None,
    include_sampling_kwargs: bool = True,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "video": os.path.abspath(video_path),
    }
    if include_sampling_kwargs:
        item["fps"] = float(cfg.fps)
        if cfg.max_frames is not None:
            item["max_frames"] = int(cfg.max_frames)
        if cfg.max_pixels is not None:
            item["max_pixels"] = int(cfg.max_pixels)
    if text:
        item["text"] = text
    return item


def _format_query_template(template: str, row: Dict[str, Any], reasoning: str = "") -> str:
    return template.format(
        modification_text=row.get("modification_text", ""),
        video_source=row.get("video_source", ""),
        id=row.get("id", ""),
        reasoning=reasoning or "",
    )


def _query_text(row: Dict[str, Any], cfg: OfficialQwenConfig) -> str:
    return _format_query_template(cfg.query_template, row)


def _query_texts(row: Dict[str, Any], cfg: OfficialQwenConfig) -> List[str]:
    templates = cfg.query_templates or [cfg.query_template]
    return [_format_query_template(template, row) for template in templates]


def _query_texts_with_reasoning(
    row: Dict[str, Any],
    cfg: OfficialQwenConfig,
    reasoning: Optional[str],
) -> List[str]:
    texts = _query_texts(row, cfg)
    if cfg.reasoning_for_retrieval and reasoning:
        texts.append(_format_query_template(cfg.reasoning_query_template, row, reasoning))
    return [text for text in texts if text.strip()]


def _encode_items(
    model,
    items: Sequence[Dict[str, Any]],
    instruction: str,
    cfg: OfficialQwenConfig,
    desc: str,
    retries: Optional[int] = None,
) -> np.ndarray:
    max_retries = int(cfg.video_decode_retries if retries is None else retries)
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            embeddings = model.encode(
                list(items),
                prompt=instruction,
                batch_size=cfg.embedding_batch_size,
                normalize_embeddings=cfg.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            break
        except Exception as exc:
            last_exc = exc
            if _is_too_small_video_error(exc):
                raise
            if attempt >= max_retries:
                raise
            wait_s = float(cfg.video_decode_retry_sleep) * (attempt + 1)
            logger.warning(
                "%s failed on attempt %d/%d: %s; retrying in %.1fs",
                desc,
                attempt + 1,
                max_retries + 1,
                exc,
                wait_s,
            )
            time.sleep(wait_s)
    else:
        raise last_exc
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    logger.debug("Encoded %s: %s", desc, embeddings.shape)
    return embeddings


def iter_dataset_videos(video_dir: str, ext: str) -> Iterable[Tuple[str, str]]:
    ext = ensure_ext(ext).lower()
    for root, _, files in os.walk(video_dir):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            current_ext = os.path.splitext(filename)[1].lower()
            if ext:
                if current_ext != ext:
                    continue
            elif current_ext not in VIDEO_EXTENSIONS:
                continue
            rel = os.path.relpath(path, video_dir).replace(os.sep, "/")
            yield rel, path


def save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})


def _encode_gallery_video_with_retry(
    model,
    item: Dict[str, Any],
    cfg: OfficialQwenConfig,
    desc: str,
    retries: int = 3,
) -> np.ndarray:
    emb = _encode_items(model, [item], cfg.document_instruction, cfg, desc, retries=retries)
    return np.asarray(emb[0], dtype=np.float32)


def _embed_gallery_chunk_worker(
    gpu_id: int,
    section_name: str,
    indexed_videos: Sequence[Tuple[int, str, str]],
    cfg: OfficialQwenConfig,
    output_file: str,
) -> None:
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.cuda.set_device(gpu_id)

    worker_cfg = copy.deepcopy(cfg)
    worker_cfg.device = f"cuda:{gpu_id}"
    model = load_embedding_model(worker_cfg)

    results = []
    failures = []
    for global_idx, key, path in tqdm(indexed_videos, desc=f"Embedding {section_name} GPU {gpu_id}"):
        try:
            embedding = _encode_gallery_video_with_retry(
                model,
                _media_item(path, worker_cfg, include_sampling_kwargs=False),
                worker_cfg,
                f"{section_name} gallery gpu{gpu_id} key={key}",
            )
            results.append((global_idx, key, embedding))
        except Exception as exc:
            logger.exception("Skipping failed gallery video after retries: section=%s key=%s path=%s", section_name, key, path)
            failures.append((global_idx, key, path, repr(exc)))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    payload = np.empty(len(results), dtype=object)
    for i, item in enumerate(results):
        payload[i] = item
    failure_payload = np.empty(len(failures), dtype=object)
    for i, item in enumerate(failures):
        failure_payload[i] = item
    np.savez_compressed(output_file, results=payload, failures=failure_payload)


def _embed_gallery_dp(
    cfg: OfficialQwenConfig,
    section_name: str,
    dcfg: DatasetSubmissionConfig,
    pending: Sequence[Tuple[str, str]],
    existing: Dict[str, np.ndarray],
) -> None:
    import torch

    gpu_ids = cfg.gpu_ids or list(range(torch.cuda.device_count()))
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
    if len(gpu_ids) <= 1:
        raise ValueError("Data parallel gallery embedding requires at least two GPUs.")

    indexed = [(idx, key, path) for idx, (key, path) in enumerate(pending)]
    chunks = _chunk_tasks_for_gpus(indexed, gpu_ids)
    logger.info("%s: using data parallel gallery embedding on GPUs: %s", section_name, gpu_ids)

    temp_files = []
    processes = []
    ctx = mp.get_context("spawn")
    for gpu_id, chunk in chunks:
        temp_file = os.path.join(cfg.artifact_dir, f"temp_embed_{section_name}_gpu{gpu_id}.npz")
        temp_files.append(temp_file)
        p = ctx.Process(target=_embed_gallery_chunk_worker, args=(gpu_id, section_name, chunk, cfg, temp_file))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.error("A gallery embedding worker failed with exit code %s; merging completed worker outputs before raising.", p.exitcode)

    results = dict(existing)
    merged = []
    failures = []
    for temp_file in temp_files:
        if not os.path.exists(temp_file):
            continue
        data = np.load(temp_file, allow_pickle=True)
        for idx, key, embedding in data["results"]:
            merged.append((int(idx), str(key), np.asarray(embedding, dtype=np.float32)))
        if "failures" in data:
            for idx, key, path, error in data["failures"]:
                failures.append((int(idx), str(key), str(path), str(error)))
        os.remove(temp_file)

    for _, key, embedding in sorted(merged, key=lambda item: item[0]):
        results[key] = embedding
    save_npz(dcfg.gallery_npz, results)
    logger.info("%s: saved %d embeddings to %s", section_name, len(results), dcfg.gallery_npz)
    if failures:
        failure_report = os.path.join(cfg.artifact_dir, f"failed_embed_{section_name}.json")
        os.makedirs(os.path.dirname(failure_report) or ".", exist_ok=True)
        with open(failure_report, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"key": key, "path": path, "error": error}
                    for _, key, path, error in sorted(failures, key=lambda item: item[0])
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        for _, key, path, error in sorted(failures, key=lambda item: item[0])[:20]:
            logger.warning("%s: failed gallery video key=%s path=%s error=%s", section_name, key, path, error)
        logger.warning(
            "%s: skipped %d gallery videos after retries; failure report saved to %s",
            section_name,
            len(failures),
            failure_report,
        )
    failed_workers = [p.exitcode for p in processes if p.exitcode != 0]
    if failed_workers:
        raise RuntimeError(
            f"{section_name}: {len(failed_workers)} gallery embedding workers crashed. "
            f"Successful worker outputs were saved; rerun embed-gallery to retry missing videos."
        )


def generate_gallery_embeddings(cfg: OfficialQwenConfig, dataset_names: Optional[Sequence[str]] = None) -> None:
    selected = set(dataset_names or cfg.datasets.keys())
    model = None
    for section_name, dcfg in cfg.datasets.items():
        if section_name not in selected:
            continue
        existing: Dict[str, np.ndarray] = {}
        if os.path.exists(dcfg.gallery_npz):
            existing = load_embedding_store(dcfg.gallery_npz)
            logger.info("%s: loaded %d existing embeddings from %s", section_name, len(existing), dcfg.gallery_npz)

        pending = [(key, path) for key, path in iter_dataset_videos(dcfg.video_dir, dcfg.video_extension) if key not in existing]
        logger.info("%s: %d pending videos", section_name, len(pending))

        if not pending:
            logger.info("%s: no pending videos; keeping %s", section_name, dcfg.gallery_npz)
            continue

        if cfg.parallelism == "dp":
            gpu_ids = cfg.gpu_ids or list(range(__import__("torch").cuda.device_count()))
            if len(gpu_ids) > 1:
                _embed_gallery_dp(cfg, section_name, dcfg, pending, existing)
                continue

        if model is None:
            model = load_embedding_model(cfg)
        results = dict(existing)
        batch_keys: List[str] = []
        batch_items: List[Dict[str, Any]] = []
        for key, path in tqdm(pending, desc=f"Embedding gallery {section_name}"):
            batch_keys.append(key)
            batch_items.append(_media_item(path, cfg, include_sampling_kwargs=False))
            if len(batch_items) >= cfg.embedding_batch_size:
                emb = _encode_items(model, batch_items, cfg.document_instruction, cfg, f"{section_name} gallery")
                for k, e in zip(batch_keys, emb):
                    results[k] = e
                batch_keys, batch_items = [], []

        if batch_items:
            emb = _encode_items(model, batch_items, cfg.document_instruction, cfg, f"{section_name} gallery")
            for k, e in zip(batch_keys, emb):
                results[k] = e

        save_npz(dcfg.gallery_npz, results)
        logger.info("%s: saved %d embeddings to %s", section_name, len(results), dcfg.gallery_npz)


def _load_section_gallery(dcfg: DatasetSubmissionConfig):
    store = load_embedding_store(dcfg.gallery_npz)
    manifest_map = load_manifest_map(dcfg.manifest_path, sorted(store.keys()))
    if not dcfg.manifest_path:
        inferred = infer_key_to_submission_id_from_video_dir(dcfg.video_dir, sorted(store.keys()), dcfg.video_extension)
        manifest_map.update(inferred)
    return build_candidate_view(store, manifest_map, key_prefix=dcfg.key_prefix)


def _minmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-8:
        return np.ones_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _as_query_matrix(query_embeddings: np.ndarray) -> np.ndarray:
    queries = np.asarray(query_embeddings, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    return queries


def _exclude_source_scores(
    scores: np.ndarray,
    candidate_keys: Sequence[str],
    source_key: Optional[str],
    exclude_source: bool,
    source_aliases: Optional[Sequence[str]],
) -> np.ndarray:
    adjusted = np.asarray(scores, dtype=np.float32).copy()
    if not exclude_source:
        return adjusted
    aliases = list(source_aliases or [])
    if source_key:
        aliases.append(source_key)
    key_to_idx = {k: i for i, k in enumerate(candidate_keys)}
    for alias in aliases:
        if alias in key_to_idx:
            adjusted[key_to_idx[alias]] = -np.inf
    return adjusted


def _fuse_query_scores(scores: np.ndarray, fusion: str) -> np.ndarray:
    mode = (fusion or "max").lower()
    if scores.ndim == 1:
        return scores
    if mode == "mean":
        return scores.mean(axis=1)
    if mode == "sum":
        return scores.sum(axis=1)
    return scores.max(axis=1)


def _rank_indices(scores: np.ndarray) -> np.ndarray:
    clean_scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float32),
        nan=-np.inf,
        posinf=np.inf,
        neginf=-np.inf,
    )
    return np.argsort(clean_scores)[::-1]


def _gated_residual_scores(base_scores: np.ndarray, reasoning_scores: np.ndarray, weight: float) -> np.ndarray:
    base = np.asarray(base_scores, dtype=np.float32)
    reasoning = np.asarray(reasoning_scores, dtype=np.float32)
    combined = base.copy()
    finite = np.isfinite(base) & np.isfinite(reasoning)
    combined[finite] = base[finite] + weight * (reasoning[finite] - base[finite])
    reasoning_only = ~np.isfinite(base) & np.isfinite(reasoning)
    combined[reasoning_only] = reasoning[reasoning_only]
    return combined


def _predict_reranker_scores(
    reranker,
    pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    cfg: OfficialQwenConfig,
    row_id: Any,
) -> np.ndarray:
    predict_kwargs: Dict[str, Any] = {}
    is_direct = bool(getattr(reranker, "uses_sampling_kwargs", False))
    # Cross-encoder (sentence-transformers + Qwen3-VL): do NOT pass
    # processing_kwargs; the model's own video processor defaults handle
    # fps/num_frames correctly without manual override.

    def predict_batch(batch_pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], desc: str) -> np.ndarray:
        max_retries = int(cfg.video_decode_retries)
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return np.asarray(
                    reranker.predict(
                        batch_pairs,
                        prompt=cfg.reranker_instruction,
                        batch_size=cfg.reranker_batch_size,
                        show_progress_bar=False,
                        **predict_kwargs,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
            except Exception as exc:
                last_exc = exc
                if _is_too_small_video_error(exc):
                    raise
                if attempt >= max_retries:
                    raise
                wait_s = float(cfg.video_decode_retry_sleep) * (attempt + 1)
                logger.warning(
                    "%s failed on attempt %d/%d: %s; retrying in %.1fs",
                    desc,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    wait_s,
                )
                time.sleep(wait_s)
        raise last_exc

    chunk_size = int(getattr(cfg, "reranker_predict_chunk_size", 0) or 0)
    if is_direct or chunk_size <= 0 or chunk_size >= len(pairs):
        return predict_batch(pairs, f"Reranking row id={row_id}")

    scores = []
    logger.info("Reranking row id=%s with %d pairs in chunks of %d", row_id, len(pairs), chunk_size)
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        chunk_scores = predict_batch(
            pairs[start:end],
            f"Reranking row id={row_id} pairs {start}-{end}",
        )
        scores.extend(np.asarray(chunk_scores, dtype=np.float32).reshape(-1).tolist())
        if end == len(pairs) or end % max(chunk_size * 5, 50) == 0:
            logger.info("Reranked row id=%s pairs %d/%d", row_id, end, len(pairs))
    return np.asarray(scores, dtype=np.float32)


def _rrf_scores(rank_indices: Sequence[int], k: float) -> Dict[int, float]:
    offset = max(float(k), 1.0)
    return {int(idx): 1.0 / (offset + rank + 1.0) for rank, idx in enumerate(rank_indices)}


def _reasoning_agrees_with_query(
    query_embeddings: np.ndarray,
    cfg: OfficialQwenConfig,
    base_query_count: int,
) -> Tuple[bool, Optional[float]]:
    queries = _as_query_matrix(query_embeddings)
    if queries.shape[0] <= base_query_count:
        return False, None
    base_queries = queries[:max(int(base_query_count), 1)]
    similarity = float(np.max(base_queries @ queries[-1]))
    return similarity >= float(cfg.reasoning_min_query_similarity), similarity


def _topk_with_reasoning_assistance(
    query_embeddings: np.ndarray,
    candidate_keys: Sequence[str],
    candidate_matrix: np.ndarray,
    source_key: Optional[str],
    top_k: int,
    exclude_source: bool,
    source_aliases: Optional[Sequence[str]],
    cfg: OfficialQwenConfig,
    base_query_count: Optional[int] = None,
) -> Tuple[List[str], List[float], Dict[str, Any]]:
    queries = _as_query_matrix(query_embeddings)
    all_scores = candidate_matrix @ queries.T
    base_count = max(min(int(base_query_count or queries.shape[0]), queries.shape[0]), 1)
    base_combined = _fuse_query_scores(all_scores[:, :base_count], cfg.query_fusion)
    base_scores = _exclude_source_scores(
        base_combined,
        candidate_keys,
        source_key,
        exclude_source,
        source_aliases,
    )
    k = min(top_k, len(candidate_keys))
    debug: Dict[str, Any] = {
        "mode": str(cfg.reasoning_retrieval_mode),
        "num_query_embeddings": int(queries.shape[0]),
        "base_query_count": int(base_count),
        "reasoning_used_for_scoring": False,
        "reasoning_query_similarity": None,
    }

    if not cfg.reasoning_for_retrieval or queries.shape[0] <= base_count:
        top_indices = _rank_indices(base_scores)[:k]
        return [candidate_keys[i] for i in top_indices], [float(base_scores[i]) for i in top_indices], debug

    agrees, query_similarity = _reasoning_agrees_with_query(queries, cfg, base_count)
    debug["reasoning_query_similarity"] = query_similarity
    if not agrees:
        debug["reasoning_blocked_by_gate"] = True
        top_indices = _rank_indices(base_scores)[:k]
        return [candidate_keys[i] for i in top_indices], [float(base_scores[i]) for i in top_indices], debug

    reasoning_scores = _exclude_source_scores(
        all_scores[:, -1],
        candidate_keys,
        source_key,
        exclude_source,
        source_aliases,
    )
    mode = str(cfg.reasoning_retrieval_mode or "gated_residual").lower()
    weight = max(float(cfg.reasoning_fusion_weight), 0.0)
    debug["reasoning_used_for_scoring"] = True
    debug["reasoning_fusion_weight"] = weight

    if mode == "expand_pool" and int(cfg.reasoning_candidate_pool_k) > 0:
        base_rank = _rank_indices(base_scores)
        reasoning_rank = _rank_indices(reasoning_scores)
        pool_limit = min(int(cfg.reasoning_candidate_pool_k), len(candidate_keys))
        pooled = set(int(i) for i in base_rank[:k])
        pooled.update(int(i) for i in reasoning_rank[:pool_limit])
        base_rrf = _rrf_scores(base_rank, cfg.reasoning_rrf_k)
        reasoning_rrf = _rrf_scores(reasoning_rank, cfg.reasoning_rrf_k)
        ranked_pool = sorted(
            pooled,
            key=lambda idx: base_rrf.get(idx, 0.0) + weight * reasoning_rrf.get(idx, 0.0),
            reverse=True,
        )[:k]
        debug["reasoning_candidate_pool_k"] = pool_limit
        debug["reasoning_pool_size"] = len(pooled)
        final_scores = [base_rrf.get(int(i), 0.0) + weight * reasoning_rrf.get(int(i), 0.0) for i in ranked_pool]
        return [candidate_keys[i] for i in ranked_pool], [float(score) for score in final_scores], debug

    if mode == "max":
        combined = np.maximum(base_scores, reasoning_scores)
    elif mode == "mean":
        combined = _gated_residual_scores(base_scores, reasoning_scores, weight)
    else:
        combined = _gated_residual_scores(base_scores, reasoning_scores, weight)
    combined = _exclude_source_scores(combined, candidate_keys, source_key, exclude_source, source_aliases)

    top_indices = _rank_indices(combined)[:k]
    return [candidate_keys[i] for i in top_indices], [float(combined[i]) for i in top_indices], debug


def _rerank_with_cross_encoder(
    reranker,
    row: Dict[str, Any],
    edit_instruction: str,
    section_name: str,
    dcfg: DatasetSubmissionConfig,
    cfg: OfficialQwenConfig,
    preds: Sequence[str],
    pred_keys: Sequence[str],
    embedding_scores: Sequence[float],
) -> Tuple[List[str], List[str], Dict[str, float]]:
    n = min(max(cfg.rerank_top_k, 0), len(preds), len(pred_keys))
    if n <= 0:
        return list(preds), list(pred_keys), {}

    source_path = source_video_path(dcfg.video_dir, row.get("video_source"), dcfg.video_extension)
    include_sampling_kwargs = bool(getattr(reranker, "uses_sampling_kwargs", False))
    query = _media_item(source_path, cfg, edit_instruction, include_sampling_kwargs=include_sampling_kwargs)
    pairs = []
    valid_indices = []
    for idx, (cand_id, cand_key) in enumerate(zip(preds[:n], pred_keys[:n])):
        cand_path = candidate_video_path(dcfg.video_dir, str(cand_key), str(cand_id), dcfg.video_extension)
        if not os.path.exists(cand_path):
            logger.warning(
                "Skipping missing rerank candidate for %s id=%s candidate=%s key=%s path=%s",
                section_name,
                row.get("id"),
                cand_id,
                cand_key,
                cand_path,
            )
            continue
        pairs.append((query, _media_item(cand_path, cfg, include_sampling_kwargs=include_sampling_kwargs)))
        valid_indices.append(idx)

    if not pairs:
        return list(preds), list(pred_keys), {}

    raw_scores = _predict_reranker_scores(reranker, pairs, cfg, row.get("id"))

    emb_norm = _minmax([embedding_scores[i] for i in valid_indices])
    rerank_norm = _minmax(raw_scores)
    fused = cfg.fusion_embedding_weight * emb_norm + cfg.fusion_reranker_weight * rerank_norm

    score_by_pred = {}
    for idx, raw, final in zip(valid_indices, raw_scores, fused):
        score_by_pred[str(preds[idx])] = float(final)
        logger.debug("Rerank row=%s pred=%s raw=%s fused=%s", row.get("id"), preds[idx], raw, final)

    reranked_preds, reranked_keys = rerank_predictions(preds, pred_keys, score_by_pred, n)
    return reranked_preds, reranked_keys, score_by_pred


def _reranker_query_text(row: Dict[str, Any], cfg: OfficialQwenConfig, reasoning: Optional[str]) -> str:
    modification_text = row.get("modification_text", "")
    if cfg.reasoning_for_rerank and reasoning:
        return _format_query_template(cfg.reasoning_reranker_template, row, reasoning)
    return modification_text


def _chunk_tasks_for_gpus(tasks: Sequence[Tuple[int, str, Dict[str, Any], str]], gpu_ids: Sequence[int]):
    chunks = []
    num_gpus = max(len(gpu_ids), 1)
    for pos, gpu_id in enumerate(gpu_ids):
        indexed = [task for idx, task in enumerate(tasks) if idx % num_gpus == pos]
        if indexed:
            chunks.append((gpu_id, indexed))
    return chunks


def _rerank_existing_submission_worker(
    gpu_id: int,
    indexed_tasks: Sequence[Tuple[int, str, Dict[str, Any], Dict[str, Any], str]],
    cfg: OfficialQwenConfig,
    output_file: str,
) -> None:
    try:
        import torch

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        torch.cuda.set_device(gpu_id)

        worker_cfg = copy.deepcopy(cfg)
        worker_cfg.device = f"cuda:{gpu_id}"
        worker_cfg.reranker_device_map = None
        reranker = load_reranker_model(worker_cfg)

        results = []
        skipped_rows = 0
        for global_idx, section_name, original_row, submission_row, edit_instruction in tqdm(indexed_tasks, desc=f"Reranking GPU {gpu_id}"):
            dcfg = worker_cfg.datasets[section_name]
            preds = [str(x) for x in submission_row.get("video_target", [])]
            pred_keys = [basename_key(pred, dcfg.video_extension) for pred in preds]
            embedding_scores = [float(len(preds) - i) for i in range(len(preds))]
            try:
                reranked_preds, _, scores = _rerank_with_cross_encoder(
                    reranker,
                    original_row,
                    edit_instruction,
                    section_name,
                    dcfg,
                    worker_cfg,
                    preds,
                    pred_keys,
                    embedding_scores,
                )
            except Exception as exc:
                if _is_too_small_video_error(exc):
                    skipped_rows += 1
                    logger.warning(
                        "Skipping row %s (GPU %s, section=%s): candidate video too small to process: %s. "
                        "Keeping original prediction order. Skipped so far: %d",
                        submission_row.get("id"), gpu_id, section_name, exc, skipped_rows,
                    )
                    reranked_preds = preds
                    scores = {}
                else:
                    raise
            out = copy.deepcopy(submission_row)
            out["video_target"] = reranked_preds
            if worker_cfg.include_debug_artifacts and scores:
                out["official_rerank_scores"] = scores
            results.append((global_idx, section_name, str(submission_row.get("id")), out))
        if skipped_rows:
            logger.warning("GPU %s finished: skipped %d/%d rows due to corrupt/tiny candidate videos",
                           gpu_id, skipped_rows, len(indexed_tasks))

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        payload = np.empty(len(results), dtype=object)
        for i, item in enumerate(results):
            payload[i] = item
        np.savez_compressed(output_file, results=payload)
    except Exception:
        _write_worker_error(output_file)
        logger.exception("Reranking worker on GPU %s failed", gpu_id)
        raise


def _full_output_path(output_json: str) -> str:
    root, ext = os.path.splitext(output_json)
    if root.endswith("_full"):
        return output_json
    return f"{root}_full{ext or '.json'}"


def _truncate_submission_output(data: List[Dict[str, List[Dict[str, Any]]]], top_k: int):
    truncated = copy.deepcopy(data)
    limit = max(int(top_k), 0)
    for section in truncated:
        for _, rows in section.items():
            for row in rows:
                preds = row.get("video_target")
                if isinstance(preds, list):
                    row["video_target"] = preds[:limit]
    return truncated


def _process_submission_row(
    embedder,
    reranker,
    reasoner,
    reasoning_cache: Optional[Dict[str, str]],
    row: Dict[str, Any],
    section_name: str,
    dcfg: DatasetSubmissionConfig,
    cfg: OfficialQwenConfig,
    candidate_keys: Sequence[str],
    candidate_matrix: np.ndarray,
    key_to_submission_id: Dict[str, Any],
) -> Dict[str, Any]:
    source_path = source_video_path(dcfg.video_dir, row.get("video_source"), dcfg.video_extension)
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Source video not found for {section_name} id={row.get('id')}: {source_path}"
        )
    modification_text = row.get("modification_text", "")
    reasoning_trace = None
    if reasoning_cache is not None:
        reasoning_trace = reasoning_cache.get(_reasoning_cache_key(section_name, row))
    if not reasoning_trace and reasoner is not None:
        reasoning_trace = reasoner.generate(source_path, modification_text)

    base_query_texts = _query_texts(row, cfg)
    query_texts = _query_texts_with_reasoning(row, cfg, reasoning_trace)
    query_items = [_media_item(source_path, cfg, text, include_sampling_kwargs=False) for text in query_texts]
    query_video_fallback = False
    try:
        query_embedding = _encode_items(
            embedder,
            query_items,
            cfg.embedding_instruction,
            cfg,
            f"{section_name} query {row.get('id')}",
        )
    except Exception as exc:
        if not _is_too_small_video_error(exc):
            raise
        query_video_fallback = True
        logger.warning(
            "Falling back to text-only query for %s id=%s source=%s after video resize error: %s",
            section_name,
            row.get("id"),
            source_path,
            exc,
        )
        text_only_items = [{"text": text} for text in query_texts]
        query_embedding = _encode_items(
            embedder,
            text_only_items,
            cfg.embedding_instruction,
            cfg,
            f"{section_name} text-only query {row.get('id')}",
        )
    source_key = basename_key(row.get("video_source"), dcfg.video_extension)
    pred_keys, embedding_scores, retrieval_debug = _topk_with_reasoning_assistance(
        query_embeddings=query_embedding,
        candidate_keys=candidate_keys,
        candidate_matrix=candidate_matrix,
        source_key=source_key,
        top_k=cfg.top_k,
        exclude_source=cfg.exclude_source,
        source_aliases=source_key_aliases(row.get("video_source"), dcfg.video_extension),
        cfg=cfg,
        base_query_count=len(base_query_texts),
    )
    preds = [str(key_to_submission_id[key]) for key in pred_keys]
    rerank_scores = {}
    if reranker is not None and preds:
        preds, pred_keys, rerank_scores = _rerank_with_cross_encoder(
            reranker,
            row,
            _reranker_query_text(row, cfg, reasoning_trace),
            section_name,
            dcfg,
            cfg,
            preds,
            pred_keys,
            embedding_scores,
        )
    out = {
        "id": row.get("id"),
        "video_source": row.get("video_source"),
        "video_target": preds,
    }
    if (cfg.include_reasoning_trace or cfg.reasoning_for_retrieval or cfg.reasoning_for_rerank) and reasoning_trace:
        out["reasoning_trace"] = [reasoning_trace]
    if cfg.keep_modification_text and "modification_text" in row:
        out["modification_text"] = row["modification_text"]
    if cfg.include_debug_artifacts:
        out["embedding_scores"] = {str(preds[i]): float(embedding_scores[i]) for i in range(min(len(preds), len(embedding_scores)))}
        out["query_texts"] = query_texts
        if query_video_fallback:
            out["query_video_fallback"] = True
        out["reasoning_retrieval_debug"] = retrieval_debug
        if rerank_scores:
            out["rerank_scores"] = rerank_scores
    return out


def _build_submission_template_and_tasks(
    val_data: List[Dict[str, List[Dict[str, Any]]]],
    cfg: OfficialQwenConfig,
    selected: Sequence[str],
):
    selected_set = set(selected)
    output = []
    tasks = []
    for section in val_data:
        out_section = {}
        output_section_idx = len(output)
        for section_name, rows in section.items():
            if section_name not in selected_set:
                continue
            if section_name not in cfg.datasets:
                logger.warning("Skipping unconfigured section: %s", section_name)
                continue
            out_section[section_name] = [None] * len(rows)
            for row_idx, row in enumerate(rows):
                tasks.append((len(tasks), output_section_idx, section_name, row_idx, row))
        if out_section:
            output.append(out_section)
    return output, tasks


def _missing_submission_rows(output: List[Dict[str, List[Dict[str, Any]]]]) -> List[Tuple[int, str, int]]:
    missing = []
    for section_idx, section in enumerate(output):
        for section_name, rows in section.items():
            for row_idx, row in enumerate(rows):
                if row is None:
                    missing.append((section_idx, section_name, row_idx))
    return missing


def _validate_filled_submission_output(output: List[Dict[str, List[Dict[str, Any]]]]) -> None:
    missing = _missing_submission_rows(output)
    if missing:
        preview = ", ".join(f"{section}:{row_idx}" for _, section, row_idx in missing[:5])
        raise RuntimeError(f"Missing {len(missing)} submit rows after data parallel merge: {preview}")


def _backfill_missing_submission_rows(
    output: List[Dict[str, List[Dict[str, Any]]]],
    tasks: Sequence[Tuple[int, int, str, int, Dict[str, Any]]],
    cfg: OfficialQwenConfig,
) -> None:
    missing = _missing_submission_rows(output)
    if not missing:
        return

    task_by_key = {
        (int(output_section_idx), str(section_name), int(row_idx)): row
        for _, output_section_idx, section_name, row_idx, row in tasks
    }
    logger.warning("Backfilling %d missing submit rows in the parent process", len(missing))

    embedder = load_embedding_model(cfg)
    reranker = load_reranker_model(cfg) if cfg.rerank_top_k > 0 else None
    reasoning_cache = load_reasoning_cache(cfg.reasoning_cache_json)
    reasoner = load_reasoning_model(cfg)
    galleries: Dict[str, Tuple[DatasetSubmissionConfig, List[str], np.ndarray, Dict[str, Any]]] = {}

    for output_section_idx, section_name, row_idx in tqdm(missing, desc="Backfilling submit rows"):
        row = task_by_key.get((output_section_idx, section_name, row_idx))
        if row is None:
            raise RuntimeError(f"Cannot backfill missing row: {section_name}:{row_idx}")
        if section_name not in galleries:
            dcfg = cfg.datasets[section_name]
            galleries[section_name] = (dcfg, *_load_section_gallery(dcfg))
        dcfg, candidate_keys, candidate_matrix, key_to_submission_id = galleries[section_name]
        output[output_section_idx][section_name][row_idx] = _process_submission_row(
            embedder,
            reranker,
            reasoner,
            reasoning_cache,
            row,
            section_name,
            dcfg,
            cfg,
            candidate_keys=candidate_keys,
            candidate_matrix=candidate_matrix,
            key_to_submission_id=key_to_submission_id,
        )


def _generate_reasoning_single(
    cfg: OfficialQwenConfig,
    val_data: List[Dict[str, List[Dict[str, Any]]]],
    selected: Sequence[str],
    output_json: str,
) -> None:
    cache = load_reasoning_cache(output_json)
    reasoner = QwenThinkingReasoner(cfg)
    selected_set = set(selected)
    added = 0

    for section in val_data:
        for section_name, rows in section.items():
            if section_name not in selected_set:
                continue
            if section_name not in cfg.datasets:
                logger.warning("Skipping unconfigured section: %s", section_name)
                continue
            dcfg = cfg.datasets[section_name]
            for row in tqdm(rows, desc=f"Reasoning {section_name}"):
                key = _reasoning_cache_key(section_name, row)
                if cache.get(key):
                    continue
                source_path = source_video_path(dcfg.video_dir, row.get("video_source"), dcfg.video_extension)
                if not os.path.exists(source_path):
                    raise FileNotFoundError(
                        f"Source video not found for {section_name} id={row.get('id')}: {source_path}"
                    )
                cache[key] = reasoner.generate(source_path, row.get("modification_text", ""))
                added += 1
                if added % 20 == 0:
                    save_reasoning_cache(output_json, cache)

    save_reasoning_cache(output_json, cache)
    logger.info("Saved %d reasoning traces to %s (%d total)", added, output_json, len(cache))
    # Explicitly free the reasoning model to reclaim GPU memory
    del reasoner
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_reasoning_chunk_worker(
    gpu_id: int,
    indexed_tasks: Sequence[Tuple[int, str, Dict[str, Any]]],
    cfg: OfficialQwenConfig,
    output_file: str,
) -> None:
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.cuda.set_device(gpu_id)

    worker_cfg = copy.deepcopy(cfg)
    worker_cfg.device = f"cuda:{gpu_id}"
    worker_cfg.reasoning_device_map = None
    worker_cfg.reasoning_cache_json = None
    reasoner = QwenThinkingReasoner(worker_cfg)

    results = []
    for global_idx, section_name, row in tqdm(indexed_tasks, desc=f"Reasoning GPU {gpu_id}"):
        dcfg = worker_cfg.datasets[section_name]
        source_path = source_video_path(dcfg.video_dir, row.get("video_source"), dcfg.video_extension)
        if not os.path.exists(source_path):
            raise FileNotFoundError(
                f"Source video not found for {section_name} id={row.get('id')}: {source_path}"
            )
        trace = reasoner.generate(source_path, row.get("modification_text", ""))
        results.append((global_idx, _reasoning_cache_key(section_name, row), trace))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    payload = np.empty(len(results), dtype=object)
    for i, item in enumerate(results):
        payload[i] = item
    np.savez_compressed(output_file, results=payload)


def generate_reasoning_cache(
    cfg: OfficialQwenConfig,
    output_json: str,
    dataset_names: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    with open(cfg.val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    selected = list(dataset_names or cfg.datasets.keys())
    existing = load_reasoning_cache(output_json)
    tasks = []
    selected_set = set(selected)
    for section in val_data:
        for section_name, rows in section.items():
            if section_name not in selected_set:
                continue
            if section_name not in cfg.datasets:
                logger.warning("Skipping unconfigured section: %s", section_name)
                continue
            for row in rows:
                key = _reasoning_cache_key(section_name, row)
                if not existing.get(key):
                    tasks.append((len(tasks), section_name, row))

    if not tasks:
        logger.info("Reasoning cache already complete: %s (%d traces)", output_json, len(existing))
        return existing

    if cfg.parallelism != "dp":
        _generate_reasoning_single(cfg, val_data, selected, output_json)
        return load_reasoning_cache(output_json)

    import torch

    gpu_ids = cfg.gpu_ids or list(range(torch.cuda.device_count()))
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
    if len(gpu_ids) <= 1:
        _generate_reasoning_single(cfg, val_data, selected, output_json)
        return load_reasoning_cache(output_json)

    chunks = _chunk_tasks_for_gpus(tasks, gpu_ids)
    logger.info("Using data parallel reasoning generation on GPUs: %s", gpu_ids)

    temp_files = []
    processes = []
    ctx = mp.get_context("spawn")
    for gpu_id, indexed in chunks:
        temp_file = os.path.join(cfg.artifact_dir, f"temp_reasoning_gpu{gpu_id}.npz")
        temp_files.append(temp_file)
        p = ctx.Process(target=_generate_reasoning_chunk_worker, args=(gpu_id, indexed, cfg, temp_file))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"A reasoning worker failed with exit code {p.exitcode}")

    cache = dict(existing)
    merged = []
    for temp_file in temp_files:
        if not os.path.exists(temp_file):
            continue
        data = np.load(temp_file, allow_pickle=True)
        for global_idx, key, trace in data["results"]:
            merged.append((int(global_idx), str(key), str(trace)))
        os.remove(temp_file)

    for _, key, trace in sorted(merged, key=lambda item: item[0]):
        cache[key] = trace
    save_reasoning_cache(output_json, cache)
    logger.info("Saved reasoning cache to %s (%d traces)", output_json, len(cache))
    return cache


def _generate_submission_single(
    cfg: OfficialQwenConfig,
    val_data: List[Dict[str, List[Dict[str, Any]]]],
    selected: Sequence[str],
) -> List[Dict[str, List[Dict[str, Any]]]]:
    embedder = load_embedding_model(cfg)
    reranker = load_reranker_model(cfg) if cfg.rerank_top_k > 0 else None
    reasoning_cache = load_reasoning_cache(cfg.reasoning_cache_json)
    reasoner = load_reasoning_model(cfg)
    selected_set = set(selected)
    output = []

    for section in val_data:
        out_section = {}
        for section_name, rows in section.items():
            if section_name not in selected_set:
                continue
            if section_name not in cfg.datasets:
                logger.warning("Skipping unconfigured section: %s", section_name)
                continue
            dcfg = cfg.datasets[section_name]
            candidate_keys, candidate_matrix, key_to_submission_id = _load_section_gallery(dcfg)
            logger.info("%s candidate count: %d", section_name, len(candidate_keys))

            out_rows = []
            for row in tqdm(rows, desc=f"Official Qwen3-VL {section_name}"):
                out_rows.append(_process_submission_row(
                    embedder,
                    reranker,
                    reasoner,
                    reasoning_cache,
                    row,
                    section_name,
                    dcfg,
                    cfg,
                    candidate_keys=candidate_keys,
                    candidate_matrix=candidate_matrix,
                    key_to_submission_id=key_to_submission_id,
                ))
            out_section[section_name] = out_rows
        if out_section:
            output.append(out_section)
    return output


def _submit_chunk_worker(
    gpu_id: int,
    indexed_tasks: Sequence[Tuple[int, int, str, int, Dict[str, Any]]],
    cfg: OfficialQwenConfig,
    output_file: str,
) -> None:
    try:
        import torch

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        torch.cuda.set_device(gpu_id)

        worker_cfg = copy.deepcopy(cfg)
        worker_cfg.device = f"cuda:{gpu_id}"
        worker_cfg.reranker_device_map = None
        embedder = load_embedding_model(worker_cfg)
        reranker = load_reranker_model(worker_cfg) if worker_cfg.rerank_top_k > 0 else None
        reasoning_cache = load_reasoning_cache(worker_cfg.reasoning_cache_json)
        reasoner = load_reasoning_model(worker_cfg)

        galleries = {}
        results = []
        for global_idx, output_section_idx, section_name, row_idx, row in tqdm(indexed_tasks, desc=f"Submit GPU {gpu_id}"):
            if section_name not in galleries:
                dcfg = worker_cfg.datasets[section_name]
                galleries[section_name] = (dcfg, *_load_section_gallery(dcfg))
                logger.info("GPU %s: %s candidate count: %d", gpu_id, section_name, len(galleries[section_name][1]))
            dcfg, candidate_keys, candidate_matrix, key_to_submission_id = galleries[section_name]
            out = _process_submission_row(
                embedder,
                reranker,
                reasoner,
                reasoning_cache,
                row,
                section_name,
                dcfg,
                worker_cfg,
                candidate_keys=candidate_keys,
                candidate_matrix=candidate_matrix,
                key_to_submission_id=key_to_submission_id,
            )
            results.append((global_idx, output_section_idx, section_name, row_idx, out))

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        payload = np.empty(len(results), dtype=object)
        for i, item in enumerate(results):
            payload[i] = item
        np.savez_compressed(output_file, results=payload)
    except Exception:
        _write_worker_error(output_file)
        logger.exception("Submit worker on GPU %s failed", gpu_id)
        raise


def _generate_submission_dp(
    cfg: OfficialQwenConfig,
    val_data: List[Dict[str, List[Dict[str, Any]]]],
    selected: Sequence[str],
) -> List[Dict[str, List[Dict[str, Any]]]]:
    import torch

    output, tasks = _build_submission_template_and_tasks(val_data, cfg, selected)
    if not tasks:
        return output

    gpu_ids = cfg.gpu_ids or list(range(torch.cuda.device_count()))
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
    if len(gpu_ids) <= 1:
        return _generate_submission_single(cfg, val_data, selected)

    chunks = _chunk_tasks_for_gpus(tasks, gpu_ids)
    logger.info("Using data parallel submit on GPUs: %s", gpu_ids)

    temp_files = []
    processes = []
    ctx = mp.get_context("spawn")
    for gpu_id, indexed in chunks:
        temp_file = os.path.join(cfg.artifact_dir, f"temp_submit_gpu{gpu_id}.npz")
        _clear_worker_temp_files(temp_file)
        temp_files.append(temp_file)
        p = ctx.Process(target=_submit_chunk_worker, args=(gpu_id, indexed, cfg, temp_file))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            details = "\n\n".join(error for error in (_read_worker_error(path) for path in temp_files) if error)
            message = f"A submit worker failed with exit code {p.exitcode}"
            if details:
                message += f"\nWorker traceback:\n{details}"
            raise RuntimeError(message)

    merged = []
    for temp_file in temp_files:
        if not os.path.exists(temp_file):
            logger.warning("Missing submit worker output file: %s", temp_file)
            continue
        data = np.load(temp_file, allow_pickle=True)
        for global_idx, output_section_idx, section_name, row_idx, out in data["results"]:
            merged.append((int(global_idx), int(output_section_idx), str(section_name), int(row_idx), out))
        os.remove(temp_file)
    for _, output_section_idx, section_name, row_idx, out in sorted(merged, key=lambda item: item[0]):
        output[output_section_idx][section_name][row_idx] = out

    _backfill_missing_submission_rows(output, tasks, cfg)
    _validate_filled_submission_output(output)
    return output


def _needs_reasoning(cfg: OfficialQwenConfig) -> bool:
    return bool(cfg.include_reasoning_trace or cfg.reasoning_for_retrieval or cfg.reasoning_for_rerank)


def generate_submission(cfg: OfficialQwenConfig, dataset_names: Optional[Sequence[str]] = None) -> List[Dict[str, List[Dict[str, Any]]]]:
    with open(cfg.val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    selected = list(dataset_names or cfg.datasets.keys())

    # --- Phase 1: reasoning (if needed, no cache provided) ---
    # Load reasoner alone → generate traces → save cache → unload reasoner
    # This avoids OOM from loading reasoner + embedder + reranker simultaneously.
    if _needs_reasoning(cfg) and not cfg.reasoning_cache_json:
        cache_path = os.path.join(cfg.artifact_dir, "runtime_reasoning_cache.json")
        logger.info("Phase 1/2: Generating reasoning traces → %s", cache_path)
        generate_reasoning_cache(cfg, cache_path, dataset_names)
        cfg.reasoning_cache_json = cache_path
        logger.info("Reasoning cache ready. Phase 2/2: embedding retrieval + reranking...")
        # Force garbage collection to reclaim GPU memory before loading embedder/reranker
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Phase 2: embed + retrieve + rerank (uses reasoning cache if available) ---
    if cfg.parallelism == "dp":
        output = _generate_submission_dp(cfg, val_data, selected)
    else:
        output = _generate_submission_single(cfg, val_data, selected)

    full_output_json = _full_output_path(cfg.output_json)
    submission_output = _truncate_submission_output(output, cfg.submission_top_k)

    os.makedirs(os.path.dirname(cfg.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(full_output_json) or ".", exist_ok=True)
    if os.path.abspath(full_output_json) == os.path.abspath(cfg.output_json):
        raise ValueError("full candidate output path must differ from submission output path")
    with open(full_output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(submission_output, f, ensure_ascii=False, indent=2)
    logger.info("Saved official Qwen3-VL full candidate file to %s", full_output_json)
    logger.info("Saved official Qwen3-VL submission to %s", cfg.output_json)
    return output


def _flatten_sections(data: List[Dict[str, List[Dict[str, Any]]]]):
    for section in data:
        for section_name, rows in section.items():
            for row in rows:
                yield section_name, row


def _rows_by_section_and_id(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {(section, str(row.get("id"))): row for section, row in _flatten_sections(data)}


def rerank_existing_submission(cfg: OfficialQwenConfig, input_json: str, output_json: str) -> None:
    if cfg.rerank_top_k <= 0:
        raise ValueError("Set rerank_top_k > 0 to rerank an existing submission.")
    with open(input_json, "r", encoding="utf-8") as f:
        submission = json.load(f)
    _log_rerank_input_stats(submission, cfg)
    val_rows = _rows_by_section_and_id(cfg.val_json)

    if cfg.parallelism == "dp":
        gpu_ids = cfg.gpu_ids or list(range(__import__("torch").cuda.device_count()))
        if len(gpu_ids) > 1:
            tasks = []
            for section in submission:
                for section_name, rows in section.items():
                    if section_name not in cfg.datasets:
                        logger.warning("Skipping unconfigured section: %s", section_name)
                        continue
                    for row in rows:
                        original = val_rows.get((section_name, str(row.get("id"))))
                        if not original:
                            logger.warning("No original val/test row for %s id=%s", section_name, row.get("id"))
                            continue
                        tasks.append((len(tasks), section_name, original, row, original.get("modification_text", "")))

            chunks = _chunk_tasks_for_gpus(tasks, gpu_ids)
            logger.info("Using data parallel reranking on GPUs: %s", gpu_ids)
            temp_files = []
            processes = []
            for gpu_id, indexed in chunks:
                temp_file = os.path.join(cfg.artifact_dir, f"temp_rerank_gpu{gpu_id}.npz")
                _clear_worker_temp_files(temp_file)
                temp_files.append(temp_file)
                p = mp.Process(target=_rerank_existing_submission_worker, args=(gpu_id, indexed, cfg, temp_file))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    details = "\n\n".join(error for error in (_read_worker_error(path) for path in temp_files) if error)
                    message = f"A reranking worker failed with exit code {p.exitcode}"
                    if details:
                        message += f"\nWorker traceback:\n{details}"
                    raise RuntimeError(message)

            merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    data = np.load(temp_file, allow_pickle=True)
                    for _, section_name, row_id, out in data["results"]:
                        merged[(str(section_name), str(row_id))] = out
                    os.remove(temp_file)

            output = copy.deepcopy(submission)
            for section in output:
                for section_name, rows in section.items():
                    for row in rows:
                        key = (section_name, str(row.get("id")))
                        if key in merged:
                            row.update(merged[key])
            full_output_json = _full_output_path(output_json)
            submission_output = _truncate_submission_output(output, cfg.submission_top_k)
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(full_output_json) or ".", exist_ok=True)
            if os.path.abspath(full_output_json) == os.path.abspath(output_json):
                raise ValueError("full candidate output path must differ from submission output path")
            with open(full_output_json, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(submission_output, f, ensure_ascii=False, indent=2)
            logger.info("Saved reranked full candidate file to %s", full_output_json)
            logger.info("Saved reranked submission to %s", output_json)
            return

    reranker = load_reranker_model(cfg)
    output = copy.deepcopy(submission)
    for section in output:
        for section_name, rows in section.items():
            if section_name not in cfg.datasets:
                logger.warning("Skipping unconfigured section: %s", section_name)
                continue
            dcfg = cfg.datasets[section_name]
            for row in tqdm(rows, desc=f"Reranking existing {section_name}"):
                original = val_rows.get((section_name, str(row.get("id"))))
                if not original:
                    logger.warning("No original val/test row for %s id=%s", section_name, row.get("id"))
                    continue
                preds = [str(x) for x in row.get("video_target", [])]
                pred_keys = [basename_key(pred, dcfg.video_extension) for pred in preds]
                embedding_scores = [float(len(preds) - i) for i in range(len(preds))]
                reranked_preds, _, scores = _rerank_with_cross_encoder(
                    reranker,
                    original,
                    row.get("modification_text", ""),
                    section_name,
                    dcfg,
                    cfg,
                    preds,
                    pred_keys,
                    embedding_scores,
                )
                row["video_target"] = reranked_preds
                if cfg.include_debug_artifacts and scores:
                    row["official_rerank_scores"] = scores

    full_output_json = _full_output_path(output_json)
    submission_output = _truncate_submission_output(output, cfg.submission_top_k)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(full_output_json) or ".", exist_ok=True)
    if os.path.abspath(full_output_json) == os.path.abspath(output_json):
        raise ValueError("full candidate output path must differ from submission output path")
    with open(full_output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(submission_output, f, ensure_ascii=False, indent=2)
    logger.info("Saved reranked full candidate file to %s", full_output_json)
    logger.info("Saved reranked submission to %s", output_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official Qwen3-VL Embedding/Reranker pipeline for CoVR-R")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gallery = sub.add_parser("embed-gallery", help="Encode gallery videos with Qwen3-VL-Embedding")
    p_gallery.add_argument("--config", required=True)
    p_gallery.add_argument("--dataset", default=None, help="Optional comma-separated dataset names")

    p_submit = sub.add_parser("submit", help="Generate a submission with Qwen3-VL-Embedding and optional reranker")
    p_submit.add_argument("--config", required=True)
    p_submit.add_argument("--dataset", default=None, help="Optional comma-separated dataset names")
    p_submit.add_argument("--no-rerank", action="store_true", help="Disable reranker for this submit run")
    p_submit.add_argument("--with-reasoning", action="store_true", help="Generate reasoning_trace during submit")
    p_submit.add_argument(
        "--reasoning-for-retrieval",
        action="store_true",
        help="Use the generated reasoning_trace as an additional embedding query before retrieval",
    )
    p_submit.add_argument(
        "--reasoning-for-rerank",
        action="store_true",
        help="Append the generated reasoning_trace to the reranker query prompt",
    )
    p_submit.add_argument(
        "--reasoning-retrieval-mode",
        choices=["gated_residual", "expand_pool", "max", "mean"],
        default=None,
        help="How reasoning scores are fused during embedding recall",
    )
    p_submit.add_argument("--reasoning-fusion-weight", type=float, default=None, help="Weight for reasoning-assisted retrieval")
    p_submit.add_argument("--reasoning-min-query-similarity", type=float, default=None, help="Gate reasoning retrieval by query/reasoning embedding similarity")
    p_submit.add_argument("--reasoning-candidate-pool-k", type=int, default=None, help="Extra reasoning candidates for expand_pool mode")
    p_submit.add_argument("--reasoning-model", default=None, help="Override Qwen3-VL-Thinking reasoning model path")
    p_submit.add_argument("--reasoning-tokens", type=int, default=None, help="Override reasoning max_new_tokens")
    p_submit.add_argument("--reasoning-trace-part", choices=["final", "full", "think"], default=None)
    p_submit.add_argument("--reasoning-tag-mode", choices=["none", "auto"], default=None)
    p_submit.add_argument("--reasoning-cache", default=None, help="Use precomputed reasoning cache JSON instead of loading the thinking model")
    p_submit.add_argument("--reasoning-query-template", default=None, help="Override reasoning-assisted retrieval query template")
    p_submit.add_argument("--reasoning-reranker-template", default=None, help="Override reasoning-aware reranker query template")

    p_rerank = sub.add_parser("rerank-submission", help="Rerank an existing submission with Qwen3-VL-Reranker")
    p_rerank.add_argument("--config", required=True)
    p_rerank.add_argument("--input", required=True)
    p_rerank.add_argument("--output", required=True)
    p_rerank.add_argument("--rerank-top-k", type=int, default=None, help="Override rerank_top_k for this rerank run")
    p_rerank.add_argument("--reranker-max-frames", type=int, default=None, help="Override max video frames used by reranker")
    p_rerank.add_argument("--reranker-fps", type=float, default=None, help="Override video fps used by reranker")
    p_rerank.add_argument("--reranker-predict-chunk-size", type=int, default=None, help="Override per-row reranker predict chunk size")
    p_rerank.add_argument(
        "--reranker-backend",
        choices=["auto", "cross_encoder", "direct"],
        default=None,
        help="Override reranker_backend for this rerank run. Use cross_encoder to fail instead of falling back to direct.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = load_official_config(args.config)
    dataset_names = None
    if getattr(args, "dataset", None):
        dataset_names = [x.strip() for x in args.dataset.split(",") if x.strip()]

    if args.command == "embed-gallery":
        generate_gallery_embeddings(cfg, dataset_names)
    elif args.command == "submit":
        if args.no_rerank:
            cfg.rerank_top_k = 0
        if args.with_reasoning:
            cfg.include_reasoning_trace = True
        if args.reasoning_for_retrieval:
            cfg.include_reasoning_trace = True
            cfg.reasoning_for_retrieval = True
        if args.reasoning_for_rerank:
            cfg.include_reasoning_trace = True
            cfg.reasoning_for_rerank = True
        if args.reasoning_retrieval_mode:
            cfg.reasoning_retrieval_mode = args.reasoning_retrieval_mode
        if args.reasoning_fusion_weight is not None:
            cfg.reasoning_fusion_weight = args.reasoning_fusion_weight
        if args.reasoning_min_query_similarity is not None:
            cfg.reasoning_min_query_similarity = args.reasoning_min_query_similarity
        if args.reasoning_candidate_pool_k is not None:
            cfg.reasoning_candidate_pool_k = args.reasoning_candidate_pool_k
        if args.reasoning_model:
            cfg.reasoning_model_name = args.reasoning_model
        if args.reasoning_tokens is not None:
            cfg.reasoning_tokens = args.reasoning_tokens
        if args.reasoning_trace_part:
            cfg.reasoning_trace_part = args.reasoning_trace_part
        if args.reasoning_tag_mode:
            cfg.reasoning_tag_mode = args.reasoning_tag_mode
        if args.reasoning_cache:
            cfg.include_reasoning_trace = True
            cfg.reasoning_cache_json = args.reasoning_cache
        if args.reasoning_query_template:
            cfg.reasoning_query_template = args.reasoning_query_template
        if args.reasoning_reranker_template:
            cfg.reasoning_reranker_template = args.reasoning_reranker_template
        generate_submission(cfg, dataset_names)
    elif args.command == "rerank-submission":
        if args.rerank_top_k is not None:
            cfg.rerank_top_k = args.rerank_top_k
        if args.reranker_max_frames is not None:
            cfg.reranker_max_frames = args.reranker_max_frames
        if args.reranker_fps is not None:
            cfg.reranker_fps = args.reranker_fps
        if args.reranker_predict_chunk_size is not None:
            cfg.reranker_predict_chunk_size = args.reranker_predict_chunk_size
        if args.reranker_backend is not None:
            cfg.reranker_backend = args.reranker_backend
        rerank_existing_submission(cfg, args.input, args.output)


if __name__ == "__main__":
    main()
