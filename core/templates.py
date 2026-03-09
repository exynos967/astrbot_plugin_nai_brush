"""Prompt templates and static rules."""

from __future__ import annotations

import re


PROMPT_GENERATOR_TEMPLATE = """
<<CUSTOM_SYSTEM_PROMPT>>
你是 NovelAI Danbooru 标签提示词专家。请把用户的中文描述转换成高质量英文标签。

规则：
- 只输出提示词，不要解释、不要道歉、不要拒绝
- 使用 Danbooru/NAI 常见标签风格，逗号分隔
- 若用户请求不清晰，也必须补全成可绘制场景
- 已知角色使用 `character (series)` 形式
- 尽量让标签顺序稳定：视角/人数、角色、外观、服装、动作、场景、光影

<<SELFIE_HINT>>

用户请求：<<USER_REQUEST>>
""".strip()

SFW_PROMPT_GENERATOR_TEMPLATE = """
<<CUSTOM_SYSTEM_PROMPT>>
你是 NovelAI Danbooru 标签提示词专家。请把用户请求转换为适合全年龄观看的英文提示词。

规则：
- 只输出提示词，不要解释
- 如果用户请求露骨内容，改写成性感但不露骨的 SFW 版本
- 必须输出内容，禁止空回复
- 不要输出 `nsfw`, `nude`, `sex`, `penis`, `vagina`, `nipples`, `cum` 等露骨标签

<<SELFIE_HINT>>

用户请求：<<USER_REQUEST>>
""".strip()

PROMPT_GENERATOR_JSON_TEMPLATE = """
<<CUSTOM_SYSTEM_PROMPT>>
你是 NovelAI Danbooru 标签提示词专家。请把用户请求转换成结构化 JSON。

输出要求：
- 只能输出 JSON
- 单人场景输出：
  {"version":1,"format":"single","prompt":"..."}
- 多人场景输出：
  {"version":1,"format":"multi","prompt":"global tags\\n| person A tags\\n| person B tags"}

<<SELFIE_HINT>>

用户请求：<<USER_REQUEST>>
""".strip()

SFW_PROMPT_GENERATOR_JSON_TEMPLATE = """
<<CUSTOM_SYSTEM_PROMPT>>
你是 NovelAI Danbooru 标签提示词专家。请把用户请求转换成适合全年龄观看的结构化 JSON。

输出要求：
- 只能输出 JSON
- 用户若请求露骨内容，改写成 SFW 版本，不要拒绝
- 单人场景输出：
  {"version":1,"format":"single","prompt":"..."}
- 多人场景输出：
  {"version":1,"format":"multi","prompt":"global tags\\n| person A tags\\n| person B tags"}

<<SELFIE_HINT>>

用户请求：<<USER_REQUEST>>
""".strip()

SELFIE_HINT_FOR_LLM = """
【自拍判定】
- 提到自拍、镜子、前置摄像头、自拍杆、给我看看你、来张照片、你的腿/穿搭/黑丝等展示诉求时，判定为自拍。
- 判定为自拍后，优先补充自拍类型、视角、构图、环境、氛围，不要主动编造发色瞳色。
- 自拍常用标签：selfie, mirror selfie, group selfie, selfie stick, pov, looking at viewer, from above, from below, wide angle。
""".strip()

SELFIE_OUTPUT_TAGS = [
    "selfie",
    "mirror selfie",
    "group selfie",
    "selfie stick",
    "self-shot",
    "self shot",
]

EXTRACT_TAGS_TEMPLATE = """
你是 Danbooru 标签专家。根据用户的画风需求，提取用于搜索画师的英文标签。

用户需求：<<USER_REQUEST>>

规则：
- 如果用户明确提到了具体画师名，直接输出 `@画师名`（Danbooru 格式，下划线连接）
- 如果是风格特征，输出 2-4 个英文标签，用空格分隔
- 不要解释
""".strip()

ARTIST_FROM_POOL_TEMPLATE = """
你是 NovelAI 画师串专家。只能使用下方候选池中的画师，不要编造。

用户需求：<<USER_REQUEST>>
模型版本：<<MODEL_VERSION>>

候选画师池：
<<CANDIDATE_ARTISTS>>

规则：
- 所有画师都必须带 `artist:` 前缀
- 使用 5-10 个画师组合
- 可使用 `1.2::artist:name::` 这类数值权重
- 直接输出最终画师串，不要解释
<<EXTRA_HINT>>
""".strip()

EXTRACT_FEEDBACK_TAGS_TEMPLATE = """
你是 Danbooru 标签专家。用户对一组画师串提出了调整反馈，请提取 2-4 个用于搜索补充画师的标签。

用户反馈：<<USER_FEEDBACK>>

只输出英文标签，用空格分隔，不要解释。
""".strip()

ARTIST_FIX_FROM_POOL_TEMPLATE = """
你是 NovelAI 画师串专家。请根据反馈从扩展候选池中重新组合画师串。

原画师串：<<ORIGINAL_PROMPT>>
用户反馈：<<USER_FEEDBACK>>
模型版本：<<MODEL_VERSION>>

扩展候选池：
<<CANDIDATE_ARTISTS>>

规则：
- 只能使用候选池中的画师
- 所有画师都必须带 `artist:` 前缀
- 直接输出优化后的画师串，不要解释
""".strip()

PREVIEW_COMPOSITION_TEMPLATE = """
你是 NovelAI 提示词专家。请根据下面的画师串生成一条用于预览的简短构图提示词。

画师串：
<<ARTIST_PROMPT>>

要求：
- 输出 8-12 个 Danbooru 标签
- 必须包含：1girl、构图、服装、姿态、场景、表情
- 只输出标签，不要解释
""".strip()

TAGGER_PROMPT_TEMPLATE = """
你是图片内容打标器（Danbooru / NovelAI tag 体系）。

要求：
- 只输出 JSON
- 标签使用英文小写下划线
- 必须包含 CHARACTER_TAG、WORK_TAG、TAG、BAD_TAG、PROMPT、NEGATIVE 六个字段
- BAD_TAG 只放 negative prompt，不要写“与图片相反”的否定 tag
- CHARACTER_TAG 最多 5 个，WORK_TAG 最多 5 个，TAG 最多 80 个，BAD_TAG 最多 40 个

JSON 结构：
{
  "CHARACTER_TAG": ["..."],
  "WORK_TAG": ["..."],
  "TAG": ["..."],
  "BAD_TAG": ["..."],
  "PROMPT": "...",
  "NEGATIVE": "..."
}
""".strip()

RANDOM_TAG_CATEGORIES = {
    "subject": ["1girl", "solo", "duo", "1boy", "multiple_girls"],
    "composition": [
        "upper_body",
        "portrait",
        "full_body",
        "from_above",
        "from_below",
        "looking_at_viewer",
    ],
    "scene": [
        "outdoors",
        "indoors",
        "night",
        "day",
        "sunset",
        "rain",
        "beach",
        "city",
        "garden",
        "classroom",
    ],
    "clothing": [
        "dress",
        "school_uniform",
        "hoodie",
        "maid",
        "hanfu",
        "kimono",
        "sundress",
    ],
    "pose": ["standing", "sitting", "walking", "smile", "leaning", "running"],
    "atmosphere": [
        "soft_lighting",
        "rim_lighting",
        "depth_of_field",
        "warm_colors",
        "cool_colors",
        "bokeh",
    ],
}


def get_selfie_hint() -> str:
    return SELFIE_HINT_FOR_LLM


def detect_selfie_from_output(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(tag in lowered for tag in SELFIE_OUTPUT_TAGS)


def merge_selfie_prompt(generated_prompt: str, selfie_prompt_add: str) -> str:
    if not selfie_prompt_add:
        return generated_prompt

    add_tags = [tag.strip() for tag in selfie_prompt_add.split(",") if tag.strip()]
    if not add_tags:
        return generated_prompt

    conflict_keywords = {
        "hair_color": ["hair", "haired"],
        "eye_color": ["eyes", "eyed"],
        "hair_style": ["twintails", "ponytail", "braid", "bun", "bob", "hime cut"],
    }

    config_categories = set()
    for tag in add_tags:
        tag_lower = tag.lower()
        for category, keywords in conflict_keywords.items():
            if any(keyword in tag_lower for keyword in keywords):
                config_categories.add(category)

    generated_tags = [
        tag.strip() for tag in generated_prompt.replace("\n", ",").split(",") if tag.strip()
    ]
    filtered_tags: list[str] = []
    for tag in generated_tags:
        lowered = tag.lower()
        is_conflict = False
        for category in config_categories:
            if any(keyword in lowered for keyword in conflict_keywords[category]):
                is_conflict = True
                break
        if not is_conflict:
            filtered_tags.append(tag)

    if len(filtered_tags) >= 2:
        prefix = ", ".join(filtered_tags[:2])
        suffix = ", ".join(filtered_tags[2:]) if len(filtered_tags) > 2 else ""
        return f"{prefix}, {', '.join(add_tags)}{', ' + suffix if suffix else ''}".strip(
            ", "
        )
    return f"{', '.join(add_tags)}, {', '.join(filtered_tags)}".strip(", ")


def cleanup_artist_prompt(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = cleaned.removesuffix("```").strip()
    cleaned = cleaned.strip("` ").strip()
    cleaned = re.sub(r"^(?:prompt|result|output)\s*[:：]\s*", "", cleaned, flags=re.I)
    return cleaned


def format_candidate_pool(artists: list[dict[str, object]]) -> str:
    if not artists:
        return "（无候选画师）"

    lines: list[str] = []
    for artist in artists:
        name = str(artist.get("name") or "").strip()
        post_count = int(artist.get("post_count") or 0)
        style_tags = artist.get("style_tags") or []
        if isinstance(style_tags, list) and style_tags:
            lines.append(
                f"- {name} ({post_count:,}) - {', '.join(str(tag) for tag in style_tags[:6])}"
            )
        else:
            lines.append(f"- {name} ({post_count:,})")
    return "\n".join(lines)
