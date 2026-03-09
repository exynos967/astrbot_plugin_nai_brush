"""Runtime session state storage."""

from __future__ import annotations

import time
from collections import deque
from typing import Iterable

from .models import RecentImageRecord, SessionContext, SessionRuntimeState


class SessionStateStore:
    """In-memory runtime state. Session overrides are intentionally ephemeral."""

    def __init__(self) -> None:
        self._states: dict[str, SessionRuntimeState] = {}

    def get(self, session: SessionContext) -> SessionRuntimeState:
        return self._states.setdefault(session.session_key, SessionRuntimeState())

    def track_image(
        self,
        session: SessionContext,
        message_id: str,
        prompt: str,
    ) -> None:
        state = self.get(session)
        state.recent_images.appendleft(
            RecentImageRecord(
                message_id=str(message_id),
                prompt=prompt,
                created_at=time.time(),
            )
        )

    def recent_images(self, session: SessionContext) -> Iterable[RecentImageRecord]:
        return tuple(self.get(session).recent_images)

    def find_recent_image(
        self,
        session: SessionContext,
        message_id: str,
    ) -> RecentImageRecord | None:
        for item in self.get(session).recent_images:
            if item.message_id == str(message_id):
                return item
        return None

    def latest_image(self, session: SessionContext) -> RecentImageRecord | None:
        state = self.get(session)
        return state.recent_images[0] if state.recent_images else None

    def prune_expired_images(
        self,
        session: SessionContext,
        max_age_seconds: float,
    ) -> None:
        state = self.get(session)
        now = time.time()
        valid = [
            item
            for item in state.recent_images
            if (now - item.created_at) <= max_age_seconds
        ]
        state.recent_images = deque(valid, maxlen=20)
