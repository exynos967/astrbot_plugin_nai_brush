"""Pure utility helpers for prompt parsing, post-processing and tagger output."""

from __future__ import annotations

import base64
import json
import re
from typing import Any


_COUNT_RE = re.compile(
    r"^(?:solo|\d+girls|\d+boys|\d+people|1girl|1boy)$",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"^year\s+\d{4}$", re.IGNORECASE)
_DATA_URL_RE = re.compile(
    r"^data:image/(?P<fmt>[a-zA-Z0-9+.-]+);base64,(?P<b64>[A-Za-z0-9+/=]+)$"
)
_PICID_RE = re.compile(r"\[picid:([0-9a-fA-F-]{8,})\]")

_CAMERA_TAGS = {
    "pov",
    "female pov",
    "looking at viewer",
    "from above",
    "from below",
    "wide angle",
    "close-up",
    "close up",
    "full body",
    "upper body",
    "lower body",
    "selfie",
    "mirror selfie",
    "group selfie",
    "selfie stick",
    "holding phone",
}

_IMAGE_FORMAT_PREFIX = {
    "jpeg": ("/9j/",),
    "png": ("iVBORw",),
    "webp": ("UklGR",),
    "gif": ("R0lGOD",),
}

_IMAGE_MAGIC = {
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG",),
    "webp": (b"RIFF",),
    "gif": (b"GIF8",),
}


def _strip_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return stripped

    inner = stripped[3:-3].strip()
    if "\n" not in inner:
        return inner.strip()

    first_line, rest = inner.split("\n", 1)
    if first_line.strip().isalpha() and len(first_line.strip()) < 15:
        return rest.strip()
    return inner.strip()


def _join_tags(tags: Any) -> str:
    if not isinstance(tags, list):
        return ""
    return ", ".join(
        value.strip() for value in tags if isinstance(value, str) and value.strip()
    ).strip()


def parse_prompt_from_structured_output(text: str) -> str | None:
    cleaned = _strip_code_fence(text).strip()
    if not cleaned:
        return None

    candidates = [cleaned]
    if '"prompt"' in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        version = payload.get("version")
        if version == 2 or (isinstance(version, int) and version >= 2):
            rendered = _render_from_v2(payload)
            if rendered:
                return rendered

        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            normalized = prompt.strip()
            if "\\n|" in normalized:
                normalized = normalized.replace("\\n", "\n")
            return normalized

    return None


def _render_from_v2(payload: dict[str, Any]) -> str | None:
    global_tags = payload.get("global")
    if not isinstance(global_tags, list):
        return None
    first_line = _join_tags(global_tags)
    if not first_line:
        return None

    people = payload.get("people")
    people = people if isinstance(people, list) else []
    format_value = str(payload.get("format") or "").strip().lower()

    valid_people: list[list[str]] = []
    for item in people:
        if isinstance(item, list):
            parts = [value.strip() for value in item if isinstance(value, str) and value.strip()]
            if parts:
                valid_people.append(parts)

    if format_value != "multi" or len(valid_people) <= 1:
        if valid_people:
            return _join_tags(global_tags + valid_people[0]) or first_line
        return first_line

    lines = [first_line]
    for item in valid_people:
        lines.append(f"| {_join_tags(item)}")
    return "\n".join(lines).strip()


def user_mentions_appearance(raw_request: str) -> bool:
    if not raw_request:
        return False

    lowered = raw_request.lower()
    cn_keys = [
        "头发",
        "发色",
        "发型",
        "长发",
        "短发",
        "双马尾",
        "马尾",
        "刘海",
        "黑发",
        "金发",
        "白发",
        "粉发",
        "蓝发",
        "红发",
        "紫发",
        "银发",
        "棕发",
        "眼睛",
        "瞳",
        "瞳色",
        "黑长直",
    ]
    if any(key in raw_request for key in cn_keys):
        return True

    en_keys = ["hair", "haired", "eyes", "eyed", "twintails", "ponytail", "bangs"]
    return any(key in lowered for key in en_keys)


def _strip_wrappers(tag: str) -> str:
    stripped = tag.strip()
    stripped = stripped.lstrip("{[(").rstrip("}])").strip()
    stripped = re.sub(r"^[+-]?\d+(?:\.\d+)?::", "", stripped).strip()
    stripped = re.sub(r"::\s*$", "", stripped).strip()
    return stripped


def remove_selfie_appearance_tags(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return prompt
    if "::" in prompt:
        return prompt

    hair_colors = {
        "black",
        "blonde",
        "brown",
        "blue",
        "pink",
        "white",
        "silver",
        "red",
        "green",
        "purple",
        "orange",
        "gray",
        "grey",
        "aqua",
        "cyan",
    }
    eye_colors = {
        "black",
        "brown",
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "gray",
        "grey",
        "golden",
        "yellow",
        "pink",
        "aqua",
        "cyan",
    }
    hair_styles = {
        "twintails",
        "twin tails",
        "ponytail",
        "side ponytail",
        "braid",
        "side braid",
        "pigtails",
        "hair bun",
        "bun",
        "bob cut",
        "hime cut",
        "bangs",
        "blunt bangs",
        "straight hair",
        "wavy hair",
        "curly hair",
        "messy hair",
    }

    def should_remove(tag: str) -> bool:
        core = re.sub(r"\s+", " ", _strip_wrappers(tag).lower()).strip()
        if "hair" in core and any(
            part in core for part in ("ribbon", "ornament", "clip", "pin", "bow", "band", "flower")
        ):
            return False

        match = re.match(r"^([a-z]+)\s+hair$", core)
        if match and match.group(1) in hair_colors:
            return True
        if re.match(r"^[a-z]+-haired$", core):
            return True
        if re.match(r"^(?:very )?(?:long|short|medium)\s+hair$", core):
            return True
        if core in hair_styles:
            return True

        match = re.match(r"^([a-z]+)\s+eyes$", core)
        if match and match.group(1) in eye_colors:
            return True
        return False

    return _process_prompt_lines(prompt, lambda tags: [tag for tag in tags if not should_remove(tag)])


def normalize_prompt_order(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return prompt

    def reorder(tags: list[str]) -> list[str]:
        counts: list[str] = []
        cameras: list[str] = []
        years: list[str] = []
        rest: list[str] = []

        for tag in tags:
            core = re.sub(r"\s+", " ", _strip_wrappers(tag)).strip().lower()
            if _YEAR_RE.match(core):
                years.append(tag)
            elif _COUNT_RE.match(core):
                counts.append(tag)
            elif core in _CAMERA_TAGS:
                cameras.append(tag)
            else:
                rest.append(tag)

        return cameras + counts + rest + years

    return _process_prompt_lines(prompt, reorder)


def _process_prompt_lines(prompt: str, transform) -> str:
    lines = prompt.split("\n")
    out_lines: list[str] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        prefix = ""
        if raw.startswith("|"):
            prefix = "|"
            raw = raw[1:].strip()
        tags = [tag.strip() for tag in raw.split(",") if tag.strip()]
        transformed = transform(tags)
        joined = ", ".join(transformed).strip()
        if prefix:
            out_lines.append(f"{prefix} {joined}".strip())
        else:
            out_lines.append(joined)
    return "\n".join(out_lines).strip()


def strip_data_url(image_data: str) -> tuple[str, str | None]:
    if not isinstance(image_data, str):
        return "", None
    stripped = image_data.strip()
    match = _DATA_URL_RE.match(stripped)
    if not match:
        return stripped, None
    fmt = (match.group("fmt") or "").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    return match.group("b64") or "", (fmt or None)


def guess_image_format_from_base64(image_base64: str) -> str:
    base64_value, fmt = strip_data_url(image_base64)
    if fmt in ("jpeg", "png", "webp", "gif"):
        return fmt

    head = base64_value[:20]
    for format_name, prefixes in _IMAGE_FORMAT_PREFIX.items():
        if any(head.startswith(prefix) for prefix in prefixes):
            return format_name

    try:
        raw = base64.b64decode(base64_value[:120], validate=False)
    except Exception:
        return "png"

    for format_name, magics in _IMAGE_MAGIC.items():
        if any(raw.startswith(magic) for magic in magics):
            return format_name
    return "png"


def extract_picids(text: str) -> list[str]:
    return [match.group(1) for match in _PICID_RE.finditer(text or "")]


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fence(text).strip()
    if not cleaned:
        return None

    try:
        payload = json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except Exception:
            return None

    return payload if isinstance(payload, dict) else None


def normalize_output(payload: dict[str, Any]) -> dict[str, list[str] | str]:
    result = {
        "CHARACTER_TAG": _normalize_tag_list(payload.get("CHARACTER_TAG")),
        "WORK_TAG": _normalize_tag_list(payload.get("WORK_TAG")),
        "TAG": _normalize_tag_list(payload.get("TAG")),
        "BAD_TAG": _normalize_tag_list(payload.get("BAD_TAG")),
    }

    prompt = str(payload.get("PROMPT") or "").strip()
    negative = str(payload.get("NEGATIVE") or "").strip()

    if not prompt:
        prompt_parts = result["CHARACTER_TAG"] + result["WORK_TAG"] + result["TAG"]
        prompt = ", ".join(prompt_parts)
    if not negative:
        negative = ", ".join(result["BAD_TAG"])

    result["PROMPT"] = prompt
    result["NEGATIVE"] = negative
    return result


def _normalize_tag_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(stripped)
    return out
