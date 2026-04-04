import json
import random
from pathlib import Path
from typing import Iterable

from simulator.config import resolve_repo_path


TRACE_JSON_KEYS = ("timestamp", "input_length", "output_length")


def traces_from_lengths(
    lengths: Iterable[tuple[int, int]],
    target_rps: float = 1.0,
    seed: int = 42,
) -> list[dict[str, int | float]]:
    random.seed(seed)
    mean_interval_ms = 1000.0 / target_rps
    current_time_ms = 0.0
    traces: list[dict[str, int | float]] = []
    for input_length, output_length in lengths:
        traces.append(
            {
                "timestamp": current_time_ms,
                "input_length": int(input_length),
                "output_length": int(output_length),
            }
        )
        current_time_ms += random.expovariate(1.0 / mean_interval_ms)
    return traces


def load_trace_json(path: str | Path) -> list[dict[str, int | float]]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    traces: list[dict[str, int | float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if not all(key in item for key in TRACE_JSON_KEYS):
            continue
        traces.append(
            {
                "timestamp": float(item["timestamp"]),
                "input_length": int(item["input_length"]),
                "output_length": int(item["output_length"]),
            }
        )
    return traces


def load_trace_json_files(paths: Iterable[str | Path]) -> list[dict[str, int | float]]:
    traces: list[dict[str, int | float]] = []
    for path in paths:
        traces.extend(load_trace_json(path))
    return traces


def get_dataset_lengths(
    dataset_name: str,
    tokenizer_name: str,
    num_samples: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    dataset_key = dataset_name.lower()
    if dataset_key in {"gsm8k", "arxiv", "arxiv_summarization", "dolly", "sharegpt"}:
        return _load_hf_dataset_lengths(dataset_key, tokenizer_name, num_samples, seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_hf_dataset_lengths(
    dataset_name: str,
    tokenizer_name: str,
    num_samples: int,
    seed: int,
) -> list[tuple[int, int]]:
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Dataset-backed end-to-end studies require `datasets`, `huggingface_hub`, and `transformers`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    random.seed(seed)

    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        return [
            (
                len(tokenizer.encode(dataset[idx]["question"], add_special_tokens=True)),
                len(tokenizer.encode(dataset[idx]["answer"], add_special_tokens=False)),
            )
            for idx in indices
        ]

    if dataset_name in {"arxiv", "arxiv_summarization"}:
        dataset = load_dataset("ccdv/arxiv-summarization", split="test")
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        return [
            (
                len(tokenizer.encode(dataset[idx]["article"], add_special_tokens=True)),
                len(tokenizer.encode(dataset[idx]["abstract"], add_special_tokens=False)),
            )
            for idx in indices
        ]

    if dataset_name == "dolly":
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        lengths: list[tuple[int, int]] = []
        for idx in indices:
            item = dataset[idx]
            instruction = item.get("instruction", "")
            context = item.get("context", "")
            response = item.get("response", "")
            prompt = instruction if not context else f"{instruction}\n{context}"
            lengths.append(
                (
                    len(tokenizer.encode(prompt, add_special_tokens=True)),
                    len(tokenizer.encode(response, add_special_tokens=False)),
                )
            )
        return lengths

    if dataset_name == "sharegpt":
        cache_dir = resolve_repo_path("simulator/.dataset_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        json_path = hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        with Path(json_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        sampled = random.sample(data, min(num_samples, len(data)))
        lengths = []
        for item in sampled:
            input_text = ""
            output_text = ""
            for turn in item.get("conversations", []):
                if turn.get("from") == "human" and not input_text:
                    input_text = turn.get("value", "")
                elif turn.get("from") == "gpt" and input_text and not output_text:
                    output_text = turn.get("value", "")
                    break
            if not input_text or not output_text:
                continue
            lengths.append(
                (
                    len(tokenizer.encode(input_text, add_special_tokens=True)),
                    len(tokenizer.encode(output_text, add_special_tokens=False)),
                )
            )
        return lengths

    raise ValueError(f"Unsupported dataset: {dataset_name}")
