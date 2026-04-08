from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_LABEL_ORDER = [
    "ineffective",
    "unnecessary",
    "pharma",
    "rushed",
    "side-effect",
    "mandatory",
    "country",
    "ingredients",
    "political",
    "none",
    "conspiracy",
    "religious",
]

LABEL_GUIDANCE = {
    "ineffective": "Vaccines do not work or have poor effectiveness.",
    "unnecessary": "Vaccines are not needed because disease is mild or avoidable.",
    "pharma": "Big Pharma, profit motives, corruption, or hidden financial agenda.",
    "rushed": "Vaccine development/approval/testing was too fast or incomplete.",
    "side-effect": "Safety harms, adverse effects, injury, or death from vaccines.",
    "mandatory": "Opposition to vaccine mandates, forced vaccination, passports.",
    "country": "Country-specific policy, nationalism, or geopolitics of vaccines.",
    "ingredients": "Claims about toxic ingredients, DNA, microchips, or composition fears.",
    "political": "General political framing not covered by pharma/country labels.",
    "none": "No anti-vaccine stance; neutral, unrelated, or pro-vaccine message.",
    "conspiracy": "Conspiracy narratives, hidden plots, population control, hoaxes.",
    "religious": "Religious reasoning or faith-based objections to vaccines.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify tweets with LLM provider and generate Kaggle submission CSV"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--test-file", type=str, default="test_preprocessed.csv")
    parser.add_argument(
        "--sample-submission", type=Path, default=Path("sample_submission.csv")
    )
    parser.add_argument("--output-file", type=Path, default=Path("llm_submission.csv"))
    parser.add_argument("--text-column", type=str, default=None)

    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "grok"],
        default="gemini",
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional override model name for the selected provider.",
    )
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--grok-model", type=str, default="grok-3-mini")
    parser.add_argument("--grok-base-url", type=str, default="https://api.x.ai/v1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--env-file", type=Path, default=None)

    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-between-calls", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=4096)

    parser.add_argument(
        "--resume-cache",
        type=Path,
        default=Path("artifacts/llm_predictions_cache.jsonl"),
    )
    parser.add_argument("--disable-resume", action="store_true")
    parser.add_argument(
        "--allow-fallback-zero",
        action="store_true",
        help=(
            "Allow writing zero vectors for rows when API calls fail after retries. "
            "Disabled by default to prevent silent bad submissions."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--log-file", type=Path, default=Path("artifacts/llm_inference.log")
    )
    return parser.parse_args()


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model:
        return args.model.strip()
    if args.provider == "grok":
        return args.grok_model.strip()
    return args.gemini_model.strip()


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("llm_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()

    provider_key_names = {
        "gemini": ["GEMINI_API_KEY"],
        "grok": ["GROK_API_KEY", "XAI_API_KEY"],
    }
    key_names = provider_key_names.get(args.provider, ["GEMINI_API_KEY"])

    for key_name in key_names:
        env_key = os.getenv(key_name)
        if env_key:
            return env_key.strip()

    env_candidates: List[Path] = []
    if args.env_file is not None:
        env_candidates.append(args.env_file)
    env_candidates.extend([Path(".env"), Path("..") / ".env"])

    for env_path in env_candidates:
        payload = parse_env_file(env_path)
        for key_name in key_names:
            value = payload.get(key_name)
            if value:
                return value.strip()

    key_hint = " or ".join(key_names)

    raise ValueError(
        f"Missing {key_hint}. Set environment variable, pass --api-key, or use --env-file."
    )


def load_label_order_from_sample(sample_submission_path: Path) -> List[str]:
    sample_df = pd.read_csv(sample_submission_path, nrows=1)
    columns = list(sample_df.columns)
    if not columns or columns[0] != "index":
        raise ValueError("sample_submission.csv must start with 'index' column")

    labels = columns[1:]
    if len(labels) != len(DEFAULT_LABEL_ORDER):
        raise ValueError(
            f"Expected {len(DEFAULT_LABEL_ORDER)} labels, found {len(labels)} in sample submission"
        )
    return labels


def choose_text_column(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(f"Requested text column not found: {preferred}")
        return preferred

    for candidate in ["tweet_clean", "tweet"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("No usable text column found. Tried: tweet_clean, tweet")


def normalize_label_key(name: str) -> str:
    return re.sub(r"[-_\s]+", "-", name.strip().lower())


def coerce_binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, (float, np.floating)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "positive", "pos"}:
            return 1
        if lowered in {"0", "false", "no", "n", "negative", "neg", ""}:
            return 0
        try:
            return 1 if float(lowered) >= 0.5 else 0
        except ValueError:
            return 0
    return 0


def strip_markdown_fences(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            lines = lines[1:-1]
            if lines and lines[0].strip().lower() in {"json", "javascript"}:
                lines = lines[1:]
            return "\n".join(lines).strip()
    return candidate


def extract_balanced_json_substring(text: str) -> str:
    starts = [idx for idx in [text.find("{"), text.find("[")] if idx != -1]
    if not starts:
        raise ValueError("No JSON object or array start token found")

    start = min(starts)
    opening = text[start]
    closing = "}" if opening == "{" else "]"

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == opening:
            depth += 1
            continue
        if ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Could not find balanced JSON payload in model response")


def parse_json_lenient(raw_text: str) -> Any:
    cleaned = strip_markdown_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    json_block = extract_balanced_json_substring(cleaned)
    return json.loads(json_block)


def labels_to_vector(payload: Any, label_order: Sequence[str]) -> List[int]:
    n_labels = len(label_order)
    vector = [0] * n_labels

    if isinstance(payload, list):
        for idx in range(min(len(payload), n_labels)):
            vector[idx] = coerce_binary(payload[idx])
        return vector

    if isinstance(payload, dict):
        normalized_payload = {
            normalize_label_key(str(key)): value for key, value in payload.items()
        }
        for idx, label in enumerate(label_order):
            key = normalize_label_key(label)
            if key in normalized_payload:
                vector[idx] = coerce_binary(normalized_payload[key])
        return vector

    raise ValueError(f"Unsupported labels payload type: {type(payload).__name__}")


def parse_predictions_payload(
    payload: Any,
    batch_row_ids: Sequence[int],
    label_order: Sequence[str],
) -> Dict[int, List[int]]:
    if isinstance(payload, dict) and "predictions" in payload:
        entries = payload["predictions"]
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        if payload and all(str(key).lstrip("-").isdigit() for key in payload.keys()):
            entries = [
                {"row_id": int(key), "labels": value}
                for key, value in payload.items()
            ]
        else:
            entries = [payload]
    else:
        raise ValueError("Unexpected prediction payload root type")

    if not isinstance(entries, list):
        raise ValueError("Predictions payload must contain a list")

    predictions: Dict[int, List[int]] = {}
    align_by_position = len(entries) == len(batch_row_ids)

    for position, entry in enumerate(entries):
        row_id: int | None = None
        labels_payload: Any

        if isinstance(entry, dict):
            for key in ["row_id", "id", "index"]:
                if key in entry:
                    row_id = int(entry[key])
                    break
            labels_payload = entry.get("labels", entry)
        elif isinstance(entry, list):
            labels_payload = entry
        else:
            raise ValueError(f"Unsupported prediction entry type: {type(entry).__name__}")

        if row_id is None:
            if not align_by_position:
                raise ValueError("Missing row_id in prediction entry")
            row_id = int(batch_row_ids[position])

        vector = labels_to_vector(labels_payload, label_order)
        predictions[row_id] = vector

    missing = [row_id for row_id in batch_row_ids if int(row_id) not in predictions]
    if missing:
        raise ValueError(f"Missing predictions for rows: {missing[:5]}")

    return predictions


def build_system_prompt(label_order: Sequence[str]) -> str:
    guideline_lines = [
        f"- {label}: {LABEL_GUIDANCE.get(label, 'Use dataset-defined meaning.')}"
        for label in label_order
    ]
    return (
        "You are an expert multi-label classifier for COVID-19 anti-vaccine tweets.\n"
        "Classify each tweet into binary labels. Multiple labels can be 1 at once.\n"
        "Do not explain. Output valid JSON only.\n"
        "Label definitions:\n"
        + "\n".join(guideline_lines)
    )


def build_user_prompt(
    batch_rows: Sequence[Tuple[int, str]],
    label_order: Sequence[str],
) -> str:
    label_block = ", ".join(label_order)
    input_payload = [
        {"row_id": int(row_id), "tweet": text}
        for row_id, text in batch_rows
    ]
    output_shape_example = {
        "predictions": [
            {
                "row_id": 0,
                "labels": {label: 0 for label in label_order},
            }
        ]
    }

    return (
        "Return EXACTLY one JSON object with this shape:\n"
        f"{json.dumps(output_shape_example, ensure_ascii=False)}\n"
        "Rules:\n"
        "1) Include one entry per input row_id.\n"
        "2) Include all labels for each row.\n"
        "3) Each label value must be integer 0 or 1.\n"
        "4) No extra keys outside the required structure.\n"
        "5) Keep row_id unchanged.\n"
        f"Label order reference: {label_block}\n"
        "Input rows (JSON):\n"
        f"{json.dumps(input_payload, ensure_ascii=False)}"
    )


def build_response_schema(label_order: Sequence[str]) -> Dict[str, Any]:
    label_properties = {
        label: {"type": "INTEGER"}
        for label in label_order
    }
    return {
        "type": "OBJECT",
        "properties": {
            "predictions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "row_id": {"type": "INTEGER"},
                        "labels": {
                            "type": "OBJECT",
                            "properties": label_properties,
                            "required": list(label_order),
                        },
                    },
                    "required": ["row_id", "labels"],
                },
            }
        },
        "required": ["predictions"],
    }


def call_gemini_generate_content(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    label_order: Sequence[str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout_seconds: float,
) -> str:
    normalized_model = model_name
    if normalized_model.startswith("models/"):
        normalized_model = normalized_model[len("models/") :]

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(normalized_model)}:generateContent?key={quote(api_key)}"
    )

    request_body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": build_response_schema(label_order),
        },
    }

    payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Gemini HTTP error {exc.code}: {error_body[:500]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Gemini connection error: {exc}") from exc

    parsed_response = json.loads(response_text)
    candidates = parsed_response.get("candidates")
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response_text[:500]}")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Gemini returned empty content parts: {response_text[:500]}")

    text_segments: List[str] = []
    for part in parts:
        segment = part.get("text")
        if isinstance(segment, str) and segment.strip():
            text_segments.append(segment)
    text = "\n".join(text_segments).strip()
    if not text:
        raise RuntimeError(f"Gemini response missing text field: {response_text[:500]}")
    return text


def call_grok_generate_content(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout_seconds: float,
    base_url: str,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    request_body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }

    payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Grok HTTP error {exc.code}: {error_body[:500]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Grok connection error: {exc}") from exc

    parsed_response = json.loads(response_text)
    choices = parsed_response.get("choices")
    if not choices:
        raise RuntimeError(f"Grok returned no choices: {response_text[:500]}")

    message = choices[0].get("message", {})
    content = message.get("content")
    text = ""

    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        segments: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            segment = part.get("text")
            if isinstance(segment, str) and segment.strip():
                segments.append(segment)
        text = "\n".join(segments).strip()

    if not text:
        raise RuntimeError(f"Grok response missing text content: {response_text[:500]}")
    return text


def call_model_generate_content(
    provider: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    label_order: Sequence[str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout_seconds: float,
    grok_base_url: str,
) -> str:
    if provider == "gemini":
        return call_gemini_generate_content(
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            label_order=label_order,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )

    if provider == "grok":
        return call_grok_generate_content(
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            base_url=grok_base_url,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def chunk_rows(rows: Sequence[Tuple[int, str]], batch_size: int) -> Iterable[List[Tuple[int, str]]]:
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def load_resume_cache(
    cache_path: Path,
    expected_label_count: int,
    logger: logging.Logger,
) -> Dict[int, List[int]]:
    cached: Dict[int, List[int]] = {}
    if not cache_path.exists():
        return cached

    with cache_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                row_id = int(payload["row_id"])
                labels = payload["labels"]
                if not isinstance(labels, list) or len(labels) != expected_label_count:
                    continue
                cached[row_id] = [coerce_binary(v) for v in labels]
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                logger.warning(
                    "Skip invalid cache line %d in %s", line_number, cache_path
                )
    return cached


def append_resume_cache(
    cache_path: Path,
    rows: Sequence[Tuple[int, List[int]]],
) -> None:
    if not rows:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for row_id, labels in rows:
            record = {"row_id": int(row_id), "labels": [int(v) for v in labels]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file)

    provider = args.provider.strip().lower()
    model_name = resolve_model_name(args)
    api_key = resolve_api_key(args)
    label_order = load_label_order_from_sample(args.sample_submission)

    sample_df = pd.read_csv(args.sample_submission)
    test_df = pd.read_csv(args.data_dir / args.test_file)
    text_column = choose_text_column(test_df, args.text_column)

    if args.limit > 0:
        sample_df = sample_df.head(args.limit).copy()

    if "index" not in sample_df.columns:
        raise ValueError("sample_submission.csv must contain index column")
    if "index" not in test_df.columns:
        raise ValueError("test file must contain index column")

    test_text_series = (
        test_df[["index", text_column]]
        .drop_duplicates(subset=["index"], keep="first")
        .set_index("index")[text_column]
    )

    row_ids = [int(row_id) for row_id in sample_df["index"].tolist()]
    missing_row_ids = [row_id for row_id in row_ids if row_id not in test_text_series.index]
    if missing_row_ids:
        raise ValueError(
            "Missing text rows for sample_submission indexes, e.g. "
            f"{missing_row_ids[:5]}"
        )

    rows: List[Tuple[int, str]] = []
    for row_id in row_ids:
        value = test_text_series.loc[row_id]
        if pd.isna(value):
            text = ""
        else:
            text = str(value)
        rows.append((row_id, text))

    logger.info("LLM provider: %s", provider)
    logger.info("Model: %s", model_name)
    logger.info("Text column: %s", text_column)
    logger.info("Labels: %s", ", ".join(label_order))
    logger.info("Total rows to predict: %d", len(rows))

    prediction_map: Dict[int, List[int]] = {}
    if args.disable_resume:
        logger.info("Resume cache disabled")
    else:
        prediction_map.update(
            load_resume_cache(args.resume_cache, len(label_order), logger)
        )
        logger.info("Loaded %d cached rows from %s", len(prediction_map), args.resume_cache)

    pending_rows = [item for item in rows if item[0] not in prediction_map]
    logger.info("Pending rows: %d", len(pending_rows))

    if not pending_rows:
        logger.info("All rows already cached, skipping API calls")

    system_prompt = build_system_prompt(label_order)
    total_retries = 0
    failed_batches = 0
    fallback_rows = 0

    total_batches = (len(pending_rows) + args.batch_size - 1) // args.batch_size
    for batch_rows in tqdm(
        chunk_rows(pending_rows, args.batch_size),
        total=total_batches,
        desc="LLM batches",
    ):
        batch_row_ids = [row_id for row_id, _ in batch_rows]
        user_prompt = build_user_prompt(batch_rows, label_order)

        batch_predictions: Dict[int, List[int]] | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                raw_text = call_model_generate_content(
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    label_order=label_order,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_output_tokens=args.max_output_tokens,
                    timeout_seconds=args.timeout_seconds,
                    grok_base_url=args.grok_base_url,
                )
                payload = parse_json_lenient(raw_text)
                batch_predictions = parse_predictions_payload(
                    payload=payload,
                    batch_row_ids=batch_row_ids,
                    label_order=label_order,
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < args.max_retries:
                    total_retries += 1
                    sleep_s = args.retry_base_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Batch retry %d/%d for rows %s: %s",
                        attempt,
                        args.max_retries,
                        batch_row_ids[:3],
                        exc,
                    )
                    time.sleep(sleep_s)
                else:
                    failed_batches += 1
                    logger.error(
                        "Batch failed after %d attempts for rows %s: %s",
                        args.max_retries,
                        batch_row_ids[:3],
                        exc,
                    )

        cache_this_batch = True
        if batch_predictions is None:
            if not args.allow_fallback_zero:
                raise RuntimeError(
                    "API batch failed after retries and --allow-fallback-zero is not set. "
                    "Stop here to avoid generating a corrupted submission."
                )

            cache_this_batch = False
            fallback_rows += len(batch_rows)
            batch_predictions = {
                row_id: [0] * len(label_order)
                for row_id in batch_row_ids
            }

        cache_rows: List[Tuple[int, List[int]]] = []
        for row_id in batch_row_ids:
            vector = batch_predictions[row_id]
            vector = [coerce_binary(v) for v in vector]
            prediction_map[row_id] = vector
            cache_rows.append((row_id, vector))

        if not args.disable_resume and cache_this_batch:
            append_resume_cache(args.resume_cache, cache_rows)

        if args.sleep_between_calls > 0:
            time.sleep(args.sleep_between_calls)

    output_matrix = np.zeros((len(rows), len(label_order)), dtype=np.int64)
    for idx, (row_id, _) in enumerate(rows):
        vector = prediction_map.get(row_id)
        if vector is None:
            vector = [0] * len(label_order)
        output_matrix[idx, :] = np.array([coerce_binary(v) for v in vector], dtype=np.int64)

    output_df = sample_df.copy()
    if list(output_df.columns[1:]) != list(label_order):
        raise ValueError("Label order mismatch between sample submission and inferred labels")

    for label_index, label in enumerate(label_order):
        output_df[label] = output_matrix[:, label_index]

    if len(output_df) != len(rows):
        raise ValueError(
            f"Output row count mismatch: expected {len(rows)}, got {len(output_df)}"
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_file, index=False)

    positive_rates = {
        label: float(output_df[label].mean()) for label in label_order
    }
    summary = {
        "provider": provider,
        "model": model_name,
        "rows": len(rows),
        "batch_size": args.batch_size,
        "allow_fallback_zero": args.allow_fallback_zero,
        "retries": total_retries,
        "failed_batches": failed_batches,
        "fallback_rows": fallback_rows,
        "positive_rates": positive_rates,
        "output_file": str(args.output_file.resolve()),
    }
    summary_path = args.output_file.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Done. Submission saved to: %s", args.output_file.resolve())
    logger.info("Summary saved to: %s", summary_path.resolve())
    logger.info(
        "Retries=%d, failed_batches=%d, fallback_rows=%d",
        total_retries,
        failed_batches,
        fallback_rows,
    )


if __name__ == "__main__":
    main()