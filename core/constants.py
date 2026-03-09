"""Plugin constants for the AstrBot NAI picture plugin."""

from __future__ import annotations

PLUGIN_NAME = "astrbot_plugin_nai_pic"
DEFAULT_NAI_ENDPOINT = "/generate"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_NEGATIVE_FILTER_TAG = "{{{{{nsfw}}}}}"
MIN_RECOMMENDED_ARTIST_POST_COUNT = 100

MODEL_ALIASES = {
    "3": "nai-diffusion-3",
    "f3": "nai-diffusion-3-furry",
    "4": "nai-diffusion-4-full",
    "4.5": "nai-diffusion-4-5-full",
}

MODEL_DISPLAY_NAMES = {
    "nai-diffusion-3": "NAI V3",
    "nai-diffusion-3-furry": "NAI V3 Furry",
    "nai-diffusion-4-full": "NAI V4",
    "nai-diffusion-4-curated": "NAI V4 Curated",
    "nai-diffusion-4-5-full": "NAI V4.5",
}

SIZE_ALIASES = {
    "竖": "832x1216",
    "竖图": "832x1216",
    "v": "832x1216",
    "portrait": "832x1216",
    "横": "1216x832",
    "横图": "1216x832",
    "h": "1216x832",
    "landscape": "1216x832",
    "方": "1024x1024",
    "方图": "1024x1024",
    "s": "1024x1024",
    "square": "1024x1024",
}

SIZE_DISPLAY_NAMES = {
    "832x1216": "竖图 832x1216",
    "1216x832": "横图 1216x832",
    "1024x1024": "方图 1024x1024",
}
