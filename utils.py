import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetSubmissionConfig:
    video_dir: str
    gallery_npz: str
    video_extension: str
    manifest_path: Optional[str] = None
    key_prefix: str = ""
    output_source_as_string: bool = False


def ensure_ext(ext: str) -> str:
    return ext if ext.startswith(".") else f".{ext}"


def basename_key(video_id: Any, ext: str) -> str:
    ext = ensure_ext(ext)
    text = str(video_id).strip()
    base = os.path.basename(text)
    root, current_ext = os.path.splitext(base)
    if current_ext:
        return base
    return f"{root}{ext}"


def source_key_aliases(video_id: Any, ext: str) -> List[str]:
    ext = ensure_ext(ext)
    text = str(video_id).strip()
    root, current_ext = os.path.splitext(text)
    rel_key = text if current_ext else f"{text}{ext}"
    base_key = basename_key(video_id, ext)
    aliases = [rel_key, base_key]
    deduped = []
    for key in aliases:
        if key not in deduped:
            deduped.append(key)
    return deduped


def source_video_path(video_dir: str, video_id: Any, ext: str) -> str:
    ext = ensure_ext(ext)
    text = str(video_id).strip()
    root, current_ext = os.path.splitext(text)
    rel = text if current_ext else f"{text}{ext}"
    path_with_subdir = os.path.join(video_dir, rel)
    if os.path.exists(path_with_subdir):
        return path_with_subdir
    fallback = os.path.join(video_dir, basename_key(video_id, ext))
    return fallback


def infer_key_to_submission_id_from_video_dir(video_dir: str, candidate_keys: Sequence[str], ext: str) -> Dict[str, str]:

    ext = ensure_ext(ext).lower()
    basename_to_submission_id: Dict[str, str] = {}
    for root, _, files in os.walk(video_dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() != ext:
                continue
            abs_path = os.path.join(root, filename)
            rel = os.path.relpath(abs_path, video_dir)
            rel_no_ext = os.path.splitext(rel)[0].replace(os.sep, "/")
            basename_to_submission_id[filename] = rel_no_ext
            basename_to_submission_id[rel.replace(os.sep, "/")] = rel_no_ext

    mapping: Dict[str, str] = {}
    for key in candidate_keys:
        norm_key = str(key).replace(os.sep, "/")
        fallback = os.path.splitext(os.path.basename(norm_key))[0]
        mapping[key] = basename_to_submission_id.get(norm_key) or basename_to_submission_id.get(os.path.basename(norm_key)) or fallback
    return mapping


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        denom = np.linalg.norm(x) + 1e-12
        return x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def load_embedding_store(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as npz:
        return {str(k): np.asarray(npz[k], dtype=np.float32) for k in npz.files}


def load_manifest_map(manifest_path: Optional[str], candidate_keys: Sequence[str]) -> Dict[str, str]:
    fallback = {k: os.path.splitext(os.path.basename(k))[0] for k in candidate_keys}
    if not manifest_path:
        return fallback

    mapping: Dict[str, str] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        if manifest_path.endswith(".jsonl"):
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            obj = json.load(f)
            if isinstance(obj, dict):
                if "items" in obj and isinstance(obj["items"], list):
                    rows = obj["items"]
                else:
                    mapping.update({str(k): str(v) for k, v in obj.items()})
                    rows = []
            else:
                rows = obj

    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("key") or row.get("npz_key") or row.get("video_key")
        sid = row.get("submission_id") or row.get("video_id") or row.get("video_source") or row.get("id")
        path = row.get("path") or row.get("video_path")
        if key is None and path:
            key = os.path.basename(str(path))
        if key is not None and sid is not None:
            mapping[str(key)] = str(sid)

    for key in candidate_keys:
        mapping.setdefault(key, fallback[key])
    return mapping


def build_candidate_view(
    embedding_store: Dict[str, np.ndarray],
    key_to_submission_id: Optional[Dict[str, str]] = None,
    key_prefix: str = "",
) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    keys = sorted(k for k in embedding_store.keys() if not key_prefix or k.startswith(key_prefix))
    if not keys:
        raise ValueError(f"No candidate embeddings found for key_prefix={key_prefix!r}")
    matrix = np.stack([embedding_store[k] for k in keys]).astype(np.float32)
    matrix = l2_normalize_np(matrix)
    if key_to_submission_id is None:
        key_to_submission_id = {k: os.path.splitext(os.path.basename(k))[0] for k in keys}
    return keys, matrix, key_to_submission_id


def candidate_video_path(video_dir: str, candidate_key: str, submission_id: str, ext: str) -> str:
    ext = ensure_ext(ext)
    candidates = []
    if candidate_key:
        candidates.append(os.path.join(video_dir, str(candidate_key)))
    if submission_id:
        sid = str(submission_id)
        root, current_ext = os.path.splitext(sid)
        rel = sid if current_ext else f"{sid}{ext}"
        candidates.append(os.path.join(video_dir, rel))
        candidates.append(os.path.join(video_dir, os.path.basename(rel)))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else os.path.join(video_dir, str(candidate_key))


def rerank_predictions(
    preds: Sequence[str],
    pred_keys: Sequence[str],
    rerank_scores: Dict[str, float],
    rerank_top_k: int,
) -> Tuple[List[str], List[str]]:
    n = min(max(rerank_top_k, 0), len(preds), len(pred_keys))
    if n <= 0:
        return list(preds), list(pred_keys)
    order = sorted(range(n), key=lambda i: rerank_scores.get(str(preds[i]), float("-inf")), reverse=True)
    rest = list(range(n, len(preds)))
    final_indices = order + rest
    return [str(preds[i]) for i in final_indices], [str(pred_keys[i]) for i in final_indices]
