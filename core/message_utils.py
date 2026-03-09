"""AstrBot-specific message helpers."""

from __future__ import annotations

import asyncio
from typing import Any

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.message_components import Image, Plain, Reply

from .utils import strip_data_url


def _resolve_call_action(event: Any):
    bot = getattr(event, "bot", None)
    api = getattr(bot, "api", None)
    call_action = getattr(api, "call_action", None)
    if callable(call_action):
        return call_action
    return None


def _unwrap_action_response(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    data = result.get("data")
    if isinstance(data, dict):
        return data
    return result


async def send_text_message(event: Any, text: str) -> None:
    await event.send(MessageChain([Plain(text)]))


def extract_reply_component(event: Any) -> Reply | None:
    for component in event.get_messages():
        if isinstance(component, Reply):
            return component
    return None


def extract_reply_message_id(event: Any) -> str | None:
    reply = extract_reply_component(event)
    if reply and getattr(reply, "id", None):
        return str(reply.id)

    raw_message = getattr(event.message_obj, "raw_message", None)
    return _deep_find_reply_id(raw_message)


def _clean_message_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _deep_find_reply_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        if str(payload.get("type") or "").strip().lower() == "reply":
            data = payload.get("data")
            if isinstance(data, dict):
                for key in ("id", "message_id", "reply_to", "reply_id"):
                    message_id = _clean_message_id(data.get(key))
                    if message_id:
                        return message_id
            else:
                message_id = _clean_message_id(data)
                if message_id:
                    return message_id

        for key in (
            "reply_to",
            "reply_to_message_id",
            "reply_message_id",
            "quote_message_id",
            "reply_id",
        ):
            message_id = _clean_message_id(payload.get(key))
            if message_id:
                return message_id
        for value in payload.values():
            hit = _deep_find_reply_id(value)
            if hit:
                return hit
        return None
    if isinstance(payload, list):
        for item in payload:
            hit = _deep_find_reply_id(item)
            if hit:
                return hit
    return None


def _normalize_image_input(value: str) -> str | None:
    stripped = (value or "").strip()
    if not stripped:
        return None
    if stripped.startswith(("http://", "https://", "file:///", "base64://")):
        return stripped
    if stripped.startswith("data:image"):
        b64, _ = strip_data_url(stripped)
        return f"base64://{b64}" if b64 else None
    return stripped


async def extract_first_reply_image_input(event: Any) -> str | None:
    reply = extract_reply_component(event)
    if reply and getattr(reply, "chain", None):
        for component in reply.chain or []:
            if isinstance(component, Image):
                image_value = getattr(component, "url", None) or getattr(
                    component, "file", None
                )
                normalized = _normalize_image_input(str(image_value or ""))
                if normalized:
                    return normalized
                try:
                    base64_value = await component.convert_to_base64()
                except Exception:
                    continue
                return f"base64://{base64_value}"

    reply_id = extract_reply_message_id(event)
    if not reply_id:
        return None
    return await extract_image_input_from_onebot(event, reply_id)


async def extract_image_input_from_onebot(event: Any, message_id: str) -> str | None:
    call_action = _resolve_call_action(event)
    if event.get_platform_name() != "aiocqhttp" or call_action is None:
        return None

    params_list = [{"message_id": message_id}, {"id": message_id}]
    if str(message_id).isdigit():
        params_list.extend(
            [{"message_id": int(message_id)}, {"id": int(message_id)}]
        )

    for params in params_list:
        try:
            result = await call_action("get_msg", **params)
        except Exception:
            continue
        payload = _unwrap_action_response(result)
        messages = payload.get("message")
        if not isinstance(messages, list):
            continue
        for item in messages:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            data = item.get("data") or {}
            if not isinstance(data, dict):
                continue
            for key in ("file", "url"):
                normalized = _normalize_image_input(str(data.get(key) or ""))
                if normalized:
                    return normalized
    return None


async def send_image_message(
    event: Any,
    image_input: str,
) -> str | None:
    normalized = _normalize_image_input(image_input)
    if not normalized:
        return None

    call_action = _resolve_call_action(event)
    if event.get_platform_name() == "aiocqhttp" and call_action is not None:
        action = "send_group_msg" if event.get_group_id() else "send_private_msg"
        payload: dict[str, Any] = {
            "message": [{"type": "image", "data": {"file": normalized}}],
        }
        if event.get_group_id():
            payload["group_id"] = int(event.get_group_id())
        else:
            payload["user_id"] = int(event.get_sender_id())

        try:
            result = await call_action(action, **payload)
        except Exception as exc:
            logger.warning(f"[nai_pic] OneBot 直发图片失败，回退 event.send: {exc}")
        else:
            payload = _unwrap_action_response(result)
            message_id = payload.get("message_id")
            if message_id not in (None, ""):
                return str(message_id)

    if normalized.startswith(("http://", "https://")):
        await event.send(MessageChain([Image.fromURL(normalized)]))
    elif normalized.startswith("base64://"):
        await event.send(MessageChain([Image.fromBase64(normalized[9:])]))
    elif normalized.startswith("file:///"):
        await event.send(MessageChain([Image(file=normalized)]))
    else:
        await event.send(MessageChain([Image(file=normalized)]))
    return None


async def delete_onebot_message(event: Any, message_id: str) -> bool:
    call_action = _resolve_call_action(event)
    if event.get_platform_name() != "aiocqhttp" or call_action is None:
        return False

    params_list = [{"message_id": message_id}, {"id": message_id}]
    if str(message_id).isdigit():
        params_list.extend(
            [{"message_id": int(message_id)}, {"id": int(message_id)}]
        )

    for params in params_list:
        try:
            await call_action("delete_msg", **params)
        except Exception:
            continue
        return True
    return False


async def sleep_and_delete(event: Any, message_id: str, delay_seconds: float) -> bool:
    await asyncio.sleep(delay_seconds)
    return await delete_onebot_message(event, message_id)
