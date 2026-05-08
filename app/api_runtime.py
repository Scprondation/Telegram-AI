from __future__ import annotations

import asyncio
import base64
import io
import logging
import random
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image, ImageDraw, ImageFont
from telethon import TelegramClient, events
from telethon.tl import functions as tl_functions
from telethon.network.connection.tcpmtproxy import (
    ConnectionTcpMTProxyIntermediate,
    ConnectionTcpMTProxyRandomizedIntermediate,
)
from telethon.tl.custom import Dialog
from telethon.tl.types import (
    Chat,
    Channel,
    DocumentAttributeFilename,
    DocumentAttributeSticker,
    DocumentEmpty,
    MessageMediaPhoto,
    User,
)

from app.config import Settings
from app.responder import ImageInput, LLMResponder, RuntimeActionRequest, StickerCandidate, StickerDecision
from app.storage import ChatMemory, DailyPlan, DiaryEntry, DiarySourceMessage, HistoryMessage, HistoryStore


LOGGER = logging.getLogger("telegram-api-assistant")


@dataclass(slots=True)
class ApiChat:
    chat_id: int
    chat_label: str
    entity: Any
    is_direct: bool
    is_group: bool


@dataclass(slots=True)
class ApiChannel:
    chat_id: int
    chat_label: str
    entity: Channel
    unread_count: int


@dataclass(slots=True)
class StickerRuntimeCandidate:
    candidate: StickerCandidate
    document: Any


@dataclass(slots=True)
class PendingIncomingEvent:
    message_id: int
    text: str
    created_at: str
    reply_to_msg_id: int | None
    image_inputs: list[ImageInput]
    sender_name: str | None


class TelegramApiAssistant:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = HistoryStore(settings.history_db_path)
        self._responder = LLMResponder(settings)
        self._client: TelegramClient | None = None
        self._diary_task: asyncio.Task[None] | None = None
        self._proactive_task: asyncio.Task[None] | None = None
        self._online_presence_task: asyncio.Task[None] | None = None
        self._last_handled_incoming_id_by_chat: dict[int, int] = {}
        self._last_read_ack_id_by_chat: dict[int, int] = {}
        self._last_read_ack_id_by_channel: dict[int, int] = {}
        self._sticker_cache: list[StickerRuntimeCandidate] = []
        self._sticker_cache_refreshed_at: datetime | None = None
        self._last_online_pulse_at: datetime | None = None
        self._last_channel_sync_at: datetime | None = None
        self._last_interactive_event_at: datetime | None = datetime.now().astimezone()
        self._pending_image_inputs_by_chat: dict[int, tuple[list[ImageInput], datetime]] = {}
        self._pending_incoming_events_by_chat: dict[int, list[PendingIncomingEvent]] = {}
        self._last_diary_markdown_day: str | None = None
        self._reply_tasks_by_chat: dict[int, asyncio.Task[None]] = {}
        self._reply_revision_by_chat: dict[int, int] = {}
        self._reply_baseline_id_by_chat: dict[int, int] = {}
        self._last_sleep_catchup_day: str | None = None
        self._last_daily_plan_date: str | None = None

    async def start(self) -> None:
        api_id = self._settings.telegram_api_id
        api_hash = self._settings.telegram_api_hash
        if not api_id or not api_hash:
            raise RuntimeError(
                "For CLIENT_MODE=api you must set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env"
            )

        client_kwargs = self._build_client_kwargs()
        self._client = TelegramClient(
            str(self._settings.telegram_session_path),
            api_id,
            api_hash,
            **client_kwargs,
        )
        await self._client.start(phone=self._settings.telegram_phone)
        self._register_event_handlers()

        me = await self._client.get_me()
        display_name = getattr(me, "first_name", None) or getattr(me, "username", None) or "user"
        LOGGER.info("Telegram API is ready")
        LOGGER.info("Signed in as: %s", display_name)
        LOGGER.info("Auto-send is %s", self._settings.auto_send)
        LOGGER.info("Diary memory is %s", self._settings.diary_enabled)
        if self._settings.vision_enabled:
            LOGGER.info("Vision model is used only for incoming photos in api mode")
        else:
            LOGGER.info("Vision/photo analysis is disabled in api mode")
        LOGGER.info("Text model: %s", self._settings.text_llm_model)
        LOGGER.info("Proactive mode is %s", self._settings.proactive_enabled)
        LOGGER.info("Channels auto-read is %s", self._settings.channels_auto_read)
        LOGGER.info("Online presence pulses are %s", self._settings.online_presence_enabled)
        LOGGER.info(
            "Group reply triggers: %s",
            ", ".join(sorted(self._settings.group_reply_triggers)) or "disabled",
        )
        if self._settings.telegram_proxy_type:
            LOGGER.info(
                "Telegram proxy: %s://%s:%s",
                self._settings.telegram_proxy_type,
                self._settings.telegram_proxy_host,
                self._settings.telegram_proxy_port,
            )

        await self._set_online_presence(False, "startup")

        if self._settings.diary_enabled:
            self._diary_task = asyncio.create_task(self._diary_loop())
        if self._settings.proactive_enabled:
            self._proactive_task = asyncio.create_task(self._proactive_loop())
        if self._settings.online_presence_enabled:
            self._online_presence_task = asyncio.create_task(self._online_presence_loop())

        try:
            await self._poll_loop()
        finally:
            if self._diary_task is not None:
                self._diary_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._diary_task
            if self._proactive_task is not None:
                self._proactive_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._proactive_task
            if self._online_presence_task is not None:
                self._online_presence_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._online_presence_task
            for task in list(self._reply_tasks_by_chat.values()):
                task.cancel()
            for task in list(self._reply_tasks_by_chat.values()):
                with suppress(asyncio.CancelledError):
                    await task
            await self._set_online_presence(False, "shutdown")
            await self._client.disconnect()

    def _register_event_handlers(self) -> None:
        assert self._client is not None

        @self._client.on(events.NewMessage(incoming=True))
        async def _on_new_message(event) -> None:
            try:
                await self._handle_incoming_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Incoming event handling failed: %s", exc)

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._refresh_daily_plan_if_needed()
                await self._maybe_process_wakeup_backlog()
                await self._sync_channels()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("API polling failed: %s", exc)
            await asyncio.sleep(self._settings.poll_interval_seconds)

    @staticmethod
    def _moscow_tz():
        try:
            return ZoneInfo("Europe/Moscow")
        except ZoneInfoNotFoundError:
            return timezone(timedelta(hours=3))

    @staticmethod
    def _now_moscow() -> datetime:
        target_tz = TelegramApiAssistant._moscow_tz()
        local_now = datetime.now().astimezone()
        offset = local_now.utcoffset()
        if offset == timedelta(hours=3):
            return local_now.astimezone(target_tz)
        return datetime.now(timezone.utc).astimezone(target_tz)

    async def _set_online_presence(self, is_online: bool, reason: str) -> None:
        if self._client is None:
            return
        with suppress(Exception):
            await self._client(tl_functions.account.UpdateStatusRequest(offline=not is_online))
            LOGGER.info("Online presence set %s%s", "on" if is_online else "off", f" | {reason}" if reason else "")

    def _is_sleep_time_moscow(self) -> bool:
        now = self._now_moscow()
        return now.hour >= 23 or now.hour < 7

    async def _refresh_daily_plan_if_needed(self) -> None:
        now = self._now_moscow()
        target_date = now.date() + timedelta(days=1) if now.hour >= 23 else now.date()
        day_key = target_date.isoformat()
        if self._last_daily_plan_date == day_key:
            return
        if self._store.get_daily_plan(day_key) is not None:
            self._last_daily_plan_date = day_key
            return

        yesterday_key = (target_date - timedelta(days=1)).isoformat()
        recent_messages = self._filter_diary_source_messages(
            self._store.get_day_messages(yesterday_key, min(80, self._settings.diary_source_limit))
        )
        yesterday_summary = "\n".join(
            f"- {item.created_at} [{item.chat_label}] {item.sender_name or item.direction}: {item.text[:100]}"
            for item in recent_messages[-18:]
        )
        try:
            summary, schedule = await self._responder.generate_daily_plan(
                day_key=day_key,
                yesterday_summary=yesterday_summary,
                social_context=self._build_social_context(exclude_chat_id=0),
            )
        except Exception as exc:
            LOGGER.exception("Daily plan generation failed: %s", exc)
            summary, schedule = self._fallback_daily_plan(day_key)
        self._store.upsert_daily_plan(
            plan_date=day_key,
            summary=summary,
            schedule=schedule,
        )
        self._last_daily_plan_date = day_key
        LOGGER.info("Daily plan generated for %s", day_key)

    def _get_previous_incoming_checkpoint(self, chat_id: int) -> int:
        if chat_id in self._last_handled_incoming_id_by_chat:
            return self._last_handled_incoming_id_by_chat[chat_id]
        if chat_id in self._last_read_ack_id_by_chat:
            return self._last_read_ack_id_by_chat[chat_id]
        return self._store.get_last_incoming_message_id(chat_id) or 0

    def _get_unread_backlog_checkpoint(self, chat_id: int) -> int:
        if chat_id in self._last_handled_incoming_id_by_chat:
            return self._last_handled_incoming_id_by_chat[chat_id]
        if chat_id in self._last_read_ack_id_by_chat:
            return self._last_read_ack_id_by_chat[chat_id]
        return 0

    @staticmethod
    def _format_moscow_timestamp(value: datetime) -> str:
        target_tz = TelegramApiAssistant._moscow_tz()
        aware_value = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return aware_value.astimezone(target_tz).strftime("%Y-%m-%d %H:%M:%S")

    async def _handle_incoming_event(self, event) -> None:
        assert self._client is not None
        message = getattr(event, "message", None)
        if message is None or getattr(message, "out", False):
            return
        chat = await self._chat_from_event(event)
        if chat is None:
            return
        if not self._is_allowed_chat(chat.chat_label):
            return

        text = self._format_message_for_storage(message)
        if not text:
            return
        sender_name = await self._resolve_sender_name(event, chat)
        self._update_behavior_profile(chat, text)
        self._update_emotional_state(chat, text)
        self._last_interactive_event_at = datetime.now().astimezone()
        LOGGER.info("Incoming event in %s: %s", chat.chat_label, text)

        latest_incoming_id = int(message.id)
        previous_latest_id = self._get_previous_incoming_checkpoint(chat.chat_id)
        latest_created_at = self._format_moscow_timestamp(message.date)
        self._store.add_message(
            chat_id=chat.chat_id,
            message_id=message.id,
            direction="incoming",
            text=text,
            chat_label=chat.chat_label,
            sender_name=sender_name,
            created_at=latest_created_at,
        )
        self._refresh_chat_memory(chat)
        if latest_incoming_id <= previous_latest_id:
            return

        latest_incoming_text = text
        if self._is_sleep_time_moscow():
            LOGGER.info(
                "Sleep mode active, left unread in %s: %s",
                chat.chat_label,
                latest_incoming_text,
            )
            return

        latest_incoming_images = await self._extract_image_inputs(message)
        if latest_incoming_images and self._is_photo_placeholder_text(latest_incoming_text):
            self._cache_pending_image_context(chat.chat_id, latest_incoming_images)
            self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
            LOGGER.info("Cached photo context in %s", chat.chat_label)
            return

        latest_reply_to = getattr(message, "reply_to", None)
        latest_reply_to_msg_id = getattr(latest_reply_to, "reply_to_msg_id", None)
        self._append_pending_incoming_event(
            chat.chat_id,
            PendingIncomingEvent(
                message_id=latest_incoming_id,
                text=latest_incoming_text,
                created_at=latest_created_at,
                reply_to_msg_id=latest_reply_to_msg_id,
                image_inputs=latest_incoming_images,
                sender_name=sender_name,
            ),
        )
        await self._mark_chat_read(chat, latest_incoming_id)
        self._schedule_reply_task(chat, baseline_id=previous_latest_id)

    @staticmethod
    def _is_photo_placeholder_text(text: str) -> bool:
        stripped = text.strip()
        return stripped == "[Photo]" or stripped.startswith("[Image]")

    def _cache_pending_image_context(self, chat_id: int, image_inputs: list[ImageInput]) -> None:
        if not image_inputs:
            return
        self._pending_image_inputs_by_chat[chat_id] = (
            image_inputs[: self._settings.max_images_per_message],
            datetime.now().astimezone(),
        )

    def _take_pending_image_context(
        self,
        chat_id: int,
        latest_incoming_text: str,
    ) -> list[ImageInput]:
        pending = self._pending_image_inputs_by_chat.get(chat_id)
        if pending is None:
            return []
        image_inputs, cached_at = pending
        if datetime.now().astimezone() - cached_at > timedelta(minutes=10):
            self._pending_image_inputs_by_chat.pop(chat_id, None)
            return []
        if not self._message_likely_relates_to_recent_photo(latest_incoming_text):
            return []
        self._pending_image_inputs_by_chat.pop(chat_id, None)
        return image_inputs

    def _append_pending_incoming_event(
        self,
        chat_id: int,
        event: PendingIncomingEvent,
    ) -> None:
        pending = [
            item
            for item in self._pending_incoming_events_by_chat.get(chat_id, [])
            if item.message_id != event.message_id
        ]
        pending.append(event)
        self._pending_incoming_events_by_chat[chat_id] = pending[-8:]

    def _pending_incoming_events_after(
        self,
        chat_id: int,
        baseline_id: int,
    ) -> list[PendingIncomingEvent]:
        pending = self._pending_incoming_events_by_chat.get(chat_id, [])
        return sorted(
            (item for item in pending if item.message_id > baseline_id),
            key=lambda item: item.message_id,
        )

    def _drop_pending_incoming_events_through(self, chat_id: int, message_id: int) -> None:
        pending = [
            item
            for item in self._pending_incoming_events_by_chat.get(chat_id, [])
            if item.message_id > message_id
        ]
        if pending:
            self._pending_incoming_events_by_chat[chat_id] = pending
        else:
            self._pending_incoming_events_by_chat.pop(chat_id, None)

    @staticmethod
    def _build_incoming_burst_text(texts: list[str]) -> str:
        cleaned = [item.strip() for item in texts if item.strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        return "Новые сообщения подряд:\n" + "\n".join(f"- {item}" for item in cleaned[-5:])

    @staticmethod
    def _should_attach_chat_snapshot(latest_incoming_text: str, burst_size: int) -> bool:
        if burst_size > 1:
            return True
        value = latest_incoming_text.strip().lower().replace("ё", "е")
        if not value:
            return False
        if len(value) <= 3:
            return True
        ambiguous_tokens = {
            "?",
            "??",
            "да",
            "нет",
            "ага",
            "угу",
            "ок",
            "окей",
            "мм",
            "мда",
            "ясно",
            "пон",
            "поняла",
            "эм",
            "ладно",
            "ау",
        }
        return value in ambiguous_tokens

    def _schedule_reply_task(self, chat: ApiChat, baseline_id: int | None = None) -> None:
        chat_id = chat.chat_id
        revision = self._reply_revision_by_chat.get(chat_id, 0) + 1
        self._reply_revision_by_chat[chat_id] = revision
        previous_task = self._reply_tasks_by_chat.get(chat_id)
        if baseline_id is None:
            baseline_id = self._get_previous_incoming_checkpoint(chat_id)
        existing_baseline = self._reply_baseline_id_by_chat.get(chat_id)
        if previous_task is not None and not previous_task.done() and existing_baseline is not None:
            baseline_id = min(existing_baseline, baseline_id)
        self._reply_baseline_id_by_chat[chat_id] = baseline_id
        if previous_task is not None and not previous_task.done():
            previous_task.cancel()
        self._reply_tasks_by_chat[chat_id] = asyncio.create_task(
            self._run_debounced_reply(chat, revision, baseline_id)
        )

    def _has_active_reply_tasks(self) -> bool:
        return any(not task.done() for task in self._reply_tasks_by_chat.values())

    def _should_prioritize_reply_latency(self) -> bool:
        if not self._settings.fast_reply_mode:
            return False
        if self._has_active_reply_tasks():
            return True
        if self._last_interactive_event_at is None:
            return False
        return datetime.now().astimezone() - self._last_interactive_event_at < timedelta(
            seconds=self._settings.fast_reply_background_cooldown_seconds
        )

    async def _run_debounced_reply(self, chat: ApiChat, revision: int, baseline_id: int) -> None:
        try:
            await asyncio.sleep(self._settings.reply_debounce_seconds)
            if self._reply_revision_by_chat.get(chat.chat_id) != revision:
                return
            handled = await self._reply_to_pending_events(
                chat,
                revision=revision,
                baseline_id=baseline_id,
            )
            if not handled:
                await self._sync_and_maybe_reply(chat, revision=revision, baseline_id=baseline_id)
        except asyncio.CancelledError:
            LOGGER.info("Cancelled stale reply generation in %s", chat.chat_label)
            raise
        finally:
            current = self._reply_tasks_by_chat.get(chat.chat_id)
            if current is asyncio.current_task():
                self._reply_tasks_by_chat.pop(chat.chat_id, None)
                self._reply_baseline_id_by_chat.pop(chat.chat_id, None)

    async def _reply_to_pending_events(
        self,
        chat: ApiChat,
        revision: int | None,
        baseline_id: int,
    ) -> bool:
        pending_events = self._pending_incoming_events_after(chat.chat_id, baseline_id)
        if not pending_events:
            return False

        latest_event = pending_events[-1]
        latest_incoming_id = latest_event.message_id
        latest_incoming_at = latest_event.created_at
        latest_reply_to_msg_id = latest_event.reply_to_msg_id
        latest_incoming_text = self._build_incoming_burst_text(
            [event.text for event in pending_events[-5:]]
        )
        latest_incoming_images: list[ImageInput] = []
        for event in pending_events:
            for image_input in event.image_inputs:
                latest_incoming_images.append(image_input)
                if len(latest_incoming_images) >= self._settings.max_images_per_message:
                    break
            if len(latest_incoming_images) >= self._settings.max_images_per_message:
                break

        LOGGER.info(
            "Buffered reply context in %s: %s message(s), latest=%s",
            chat.chat_label,
            len(pending_events),
            latest_incoming_text,
        )

        if self._is_sleep_time_moscow():
            LOGGER.info(
                "Sleep mode active, left unread in %s: %s",
                chat.chat_label,
                latest_incoming_text,
            )
            return True

        group_triggered = False
        if chat.is_group:
            group_triggered = await self._should_reply_in_group(
                chat,
                latest_incoming_text,
                latest_reply_to_msg_id,
            )
            if not group_triggered:
                self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
                self._drop_pending_incoming_events_through(chat.chat_id, latest_incoming_id)
                LOGGER.info(
                    "Ignored group message in %s without relevant context: %s",
                    chat.chat_label,
                    latest_incoming_text,
                )
                return True
            latest_incoming_text = self._strip_group_trigger_prefix(latest_incoming_text)

        if self._settings.trigger_prefix:
            if not latest_incoming_text.startswith(self._settings.trigger_prefix):
                LOGGER.info(
                    "Latest message in %s ignored because prefix is missing",
                    chat.chat_label,
                )
                self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
                self._drop_pending_incoming_events_through(chat.chat_id, latest_incoming_id)
                return True
            stripped = latest_incoming_text[len(self._settings.trigger_prefix) :].strip()
            if stripped:
                latest_incoming_text = stripped

        if not latest_incoming_images:
            latest_incoming_images = self._take_pending_image_context(chat.chat_id, latest_incoming_text)
            if latest_incoming_images:
                LOGGER.info("Attached cached photo context in %s", chat.chat_label)

        history = self._store.get_recent_messages(chat.chat_id, max(24, self._settings.chat_history_limit))
        if self._should_attach_chat_snapshot(latest_incoming_text, len(pending_events)):
            snapshot_input = self._build_chat_snapshot_image_input(
                chat_label=chat.chat_label,
                history=history,
                latest_incoming_text=latest_incoming_text,
            )
            if snapshot_input is not None:
                latest_incoming_images = [snapshot_input, *latest_incoming_images][: self._settings.max_images_per_message + 1]

        gallery_reference = self._load_gallery_reference_image_input(latest_incoming_text)
        if gallery_reference is not None:
            latest_incoming_images = [*latest_incoming_images, gallery_reference][: self._settings.max_images_per_message + 2]

        diary_entries = self._select_diary_context(chat.chat_id)
        force_reply = self._is_forced_reply_chat(chat) or group_triggered
        profile = self._store.get_user_profile(chat.chat_id, chat.chat_label)
        gallery_photo_path = self._pick_gallery_photo_for_send(chat, latest_incoming_text, profile.rating)
        weather_context = await self._responder.get_moscow_weather_context(latest_incoming_text)
        action_request = await self._decide_runtime_action(chat, latest_incoming_text, history)
        reply_messages, sticker_decision = await self._generate_reply_bundle(
            chat=chat,
            latest_incoming_text=latest_incoming_text,
            history=history,
            diary_entries=diary_entries,
            image_inputs=latest_incoming_images,
            force_reply=force_reply,
            source_message_id=latest_incoming_id,
            latest_incoming_at=latest_incoming_at,
            weather_context=weather_context,
        )
        if revision is not None and self._reply_revision_by_chat.get(chat.chat_id) != revision:
            LOGGER.info("Dropped stale buffered reply result in %s", chat.chat_label)
            return True
        if not reply_messages and not sticker_decision.should_send and not gallery_photo_path and not action_request.should_run:
            self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
            self._drop_pending_incoming_events_through(chat.chat_id, latest_incoming_id)
            LOGGER.info("No AI reply produced in %s for: %s", chat.chat_label, latest_incoming_text)
            return True

        if action_request.should_run:
            await self._execute_runtime_action(chat, action_request)
        self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
        self._drop_pending_incoming_events_through(chat.chat_id, latest_incoming_id)
        await self._send_reply(
            chat,
            reply_messages,
            latest_incoming_text,
            sticker_decision,
            source_message_id=latest_incoming_id,
            gallery_photo_path=gallery_photo_path,
            revision=revision,
        )
        return True

    @staticmethod
    def _message_likely_relates_to_recent_photo(text: str) -> bool:
        value = text.strip().lower().replace("ё", "е")
        if not value:
            return False
        keywords = (
            "фото",
            "фотк",
            "картин",
            "изображ",
            "на фото",
            "на фотке",
            "что там",
            "что тут",
            "что на ней",
            "что на нем",
            "что на этом",
            "посмотри",
            "глянь",
            "разглядывать",
            "умеешь разглядывать",
        )
        return any(token in value for token in keywords)

    async def _should_reply_in_group(
        self,
        chat: ApiChat,
        text: str,
        reply_to_msg_id: int | None = None,
    ) -> bool:
        if self._matches_group_reply_trigger(text):
            return True
        if reply_to_msg_id:
            direction = self._store.get_message_direction(chat.chat_id, int(reply_to_msg_id))
            if direction == "outgoing":
                return True

        lowered = text.lower().replace("ё", "е")
        if any(token in lowered for token in ("ты", "тебе", "твой", "твоя", "твои", "ответь", "что думаешь", "чё думаешь", "ау")):
            recent_history = self._store.get_recent_messages(chat.chat_id, 8)
            recent_outgoing = sum(1 for item in recent_history[-4:] if item.role == "assistant")
            if recent_outgoing > 0:
                return True
        history = self._store.get_recent_messages(chat.chat_id, 10)
        if not history:
            return False
        decision = await self._responder.decide_group_reply(
            chat_label=chat.chat_label,
            latest_incoming_text=text,
            history=history,
        )
        if decision.should_reply:
            LOGGER.info(
                "Group context decision: reply in %s%s",
                chat.chat_label,
                f" | {decision.reason}" if decision.reason else "",
            )
            return True
        return False

    def _update_behavior_profile(self, chat: ApiChat, text: str) -> None:
        if self._is_forced_reply_chat(chat):
            self._store.set_user_profile(
                chat_id=chat.chat_id,
                chat_label=chat.chat_label,
                rating=100,
                notes="partner / highest trust",
            )
            return
        lowered = text.lower().replace("ё", "е")
        rating_delta = 0
        notes = None
        if any(token in lowered for token in ("спасибо", "люблю", "нравишься", "умница", "милая", "хорошая")):
            rating_delta += 2
            notes = "kind / warm"
        if any(token in lowered for token in ("сука", "туп", "иди нах", "пизд", "ебан", "шлюх", "дура")):
            rating_delta -= 6
            notes = "rude / insulting"
        elif any(token in lowered for token in ("ответь", "ау", "чего молчим", "игнор", "ты тут")):
            rating_delta -= 2
            notes = "pushy / impatient"
        elif any(token in lowered for token in ("секс", "нюд", "голая", "скинь", "интим")) and not self._is_forced_reply_chat(chat):
            rating_delta -= 5
            notes = "sexual / boundary issue"
        self._store.update_user_profile(
            chat_id=chat.chat_id,
            chat_label=chat.chat_label,
            rating_delta=rating_delta,
            notes=notes,
        )

    def _update_emotional_state(self, chat: ApiChat, text: str) -> None:
        lowered = text.lower().replace("ё", "е")
        is_vlad = self._is_forced_reply_chat(chat)

        warm_tokens = (
            "люблю",
            "милая",
            "моя",
            "зай",
            "скуч",
            "обним",
            "спасибо",
            "хорошая",
            "умница",
            "нравишься",
        )
        rude_tokens = (
            "иди нах",
            "нахуй",
            "туп",
            "дура",
            "сука",
            "пизд",
            "ебан",
            "шлюх",
            "урод",
        )
        pushy_tokens = (
            "ау",
            "ответь",
            "молчишь",
            "ты тут",
            "игнор",
            "где ты",
        )
        sexual_tokens = (
            "секс",
            "нюд",
            "скинь",
            "сис",
            "интим",
            "пошл",
            "18+",
        )
        jealousy_tokens = (
            "девуш",
            "бывш",
            "другая",
            "подруга",
            "флирт",
            "красивая",
        )
        apology_tokens = (
            "прости",
            "сорри",
            "извини",
            "не хотел",
        )

        affection_delta = 0
        trust_delta = 0
        loyalty_delta = 0
        attachment_delta = 0
        jealousy_delta = 0
        irritation_delta = 0
        mood: str | None = None

        if is_vlad:
            affection_delta += 1
            trust_delta += 1
            loyalty_delta += 1
            attachment_delta += 1

        if any(token in lowered for token in warm_tokens):
            affection_delta += 6 if is_vlad else 3
            trust_delta += 3
            attachment_delta += 4 if is_vlad else 1
            mood = "warm / affectionate"
        if any(token in lowered for token in apology_tokens):
            trust_delta += 2
            irritation_delta -= 7
            mood = "softening"
        if any(token in lowered for token in rude_tokens):
            trust_delta -= 5
            affection_delta -= 3
            irritation_delta += 12
            mood = "hurt / guarded"
        elif any(token in lowered for token in pushy_tokens):
            trust_delta -= 1
            irritation_delta += 4
            mood = "slightly pressured"
        if any(token in lowered for token in sexual_tokens) and not is_vlad:
            trust_delta -= 4
            affection_delta -= 2
            irritation_delta += 10
            mood = "guarded / boundary"
        if is_vlad and any(token in lowered for token in jealousy_tokens):
            jealousy_delta += 5
            attachment_delta += 2
            mood = "jealous / attached"

        if not any(
            (
                affection_delta,
                trust_delta,
                loyalty_delta,
                attachment_delta,
                jealousy_delta,
                irritation_delta,
            )
        ):
            return

        memory = self._store.update_chat_emotions(
            chat_id=chat.chat_id,
            chat_label=chat.chat_label,
            affection_delta=affection_delta,
            trust_delta=trust_delta,
            loyalty_delta=loyalty_delta,
            attachment_delta=attachment_delta,
            jealousy_delta=jealousy_delta,
            irritation_delta=irritation_delta,
            mood=mood,
        )
        LOGGER.info(
            "Emotional state updated for %s: affection=%s trust=%s loyalty=%s attachment=%s jealousy=%s irritation=%s mood=%s",
            chat.chat_label,
            memory.affection,
            memory.trust,
            memory.loyalty,
            memory.attachment,
            memory.jealousy,
            memory.irritation,
            memory.mood,
        )

    def _refresh_chat_memory(self, chat: ApiChat) -> None:
        message_count = self._store.get_message_count_for_chat(chat.chat_id)
        if message_count <= 0:
            return
        current = self._store.get_chat_memory(chat.chat_id, chat.chat_label)
        if message_count <= current.source_message_count and current.chat_label == chat.chat_label:
            return

        history_limit = max(40, self._settings.chat_history_limit)
        history = self._store.get_recent_messages(chat.chat_id, history_limit)
        incoming_texts = [
            self._strip_sender_prefix(item.content)
            for item in history
            if item.role == "user"
        ]
        memory_texts = self._select_memory_texts(chat, incoming_texts)
        inferred_gender = self._infer_chat_memory_gender(
            chat,
            memory_texts or incoming_texts,
            current.inferred_gender,
        )
        known_facts = self._merge_memory_lines(
            current.known_facts,
            self._extract_memory_facts(chat, memory_texts or incoming_texts, inferred_gender),
            limit=18,
        )
        communication_style = self._build_memory_communication_style(chat, memory_texts)
        relationship_summary = self._build_memory_relationship_summary(
            chat,
            inferred_gender,
            communication_style,
        )
        recent_events = self._merge_memory_lines("", self._extract_memory_events(memory_texts), limit=10)
        self._store.upsert_chat_memory(
            chat_id=chat.chat_id,
            chat_label=chat.chat_label,
            inferred_gender=inferred_gender,
            known_facts=known_facts,
            communication_style=communication_style,
            relationship_summary=relationship_summary,
            recent_events=recent_events,
            source_message_count=message_count,
        )
        LOGGER.info("Chat memory updated for %s (%s messages)", chat.chat_label, message_count)

    @staticmethod
    def _strip_sender_prefix(text: str) -> str:
        value = (text or "").strip()
        if ":" not in value:
            return value
        prefix, rest = value.split(":", 1)
        if len(prefix.strip()) <= 40 and rest.strip():
            return rest.strip()
        return value

    def _select_memory_texts(self, chat: ApiChat, incoming_texts: list[str]) -> list[str]:
        selected: list[str] = []
        for text in incoming_texts[-80:]:
            clean = (text or "").strip()
            if not clean:
                continue
            if self._looks_like_news_message_text(clean):
                continue
            if self._is_tender_message_text(clean) or self._looks_like_personal_memory_text(clean):
                selected.append(clean)
        if selected:
            return selected[-36:]
        if chat.is_group:
            return []
        fallback: list[str] = []
        for text in incoming_texts[-24:]:
            clean = (text or "").strip()
            if not clean:
                continue
            if self._looks_like_news_message_text(clean):
                continue
            if len(clean) < 5 or len(clean) > 140:
                continue
            if self._looks_like_plain_small_talk(clean):
                continue
            if "?" in clean or self._looks_like_personal_memory_text(clean):
                fallback.append(clean)
        return fallback[-18:]

    @staticmethod
    def _is_tender_message_text(text: str) -> bool:
        lowered = (text or "").lower().replace("ё", "е")
        tender_tokens = (
            "люблю",
            "любим",
            "неж",
            "мила",
            "милый",
            "зай",
            "котик",
            "котен",
            "солныш",
            "родная",
            "родной",
            "скучаю",
            "обнима",
            "целую",
            "лап",
            "моя девочка",
            "мой мальчик",
        )
        return any(token in lowered for token in tender_tokens)

    def _looks_like_personal_memory_text(self, text: str) -> bool:
        lowered = (text or "").lower().replace("ё", "е")
        if self._looks_like_news_message_text(text):
            return False
        personal_tokens = (
            "меня зовут",
            "зови меня",
            "мне ",
            "я из",
            "живу в",
            "город",
            "расстал",
            "отношен",
            "девуш",
            "парень",
            "влюб",
            "нравишь",
            "ревн",
            "обид",
            "боле",
            "забол",
            "экзам",
            "огэ",
            "егэ",
            "учеб",
            "школ",
            "работ",
            "устал",
            "устала",
            "спать",
            "сонный",
            "не спал",
            "поссор",
            "мама",
            "папа",
            "брат",
            "сестра",
            "семья",
        )
        if any(token in lowered for token in personal_tokens):
            return True
        if "у меня" in lowered or "со мной" in lowered:
            return True
        return False

    def _looks_like_news_message_text(self, text: str) -> bool:
        lowered = (text or "").strip().lower().replace("ё", "е")
        if not lowered:
            return False
        if self._looks_like_service_message(lowered):
            return True
        if lowered.startswith("[photo]"):
            return False
        news_tokens = (
            "http://",
            "https://",
            "t.me/",
            "telegram.me/",
            "новост",
            "канал",
            "подпис",
            "розыгрыш",
            "скидк",
            "промокод",
            "анонс",
            "обновлен",
            "обновление",
            "утекла",
            "обложк",
            "альбом",
            "update",
            "patch",
            "release",
            "матч",
            "турнир",
            "стрим",
            "результат",
            "ссылка",
            "репост",
            "опубликов",
            "голосован",
            "донат",
        )
        if any(token in lowered for token in news_tokens):
            return True
        if len(lowered) > 180 and not re.search(r"\b(?:я|мне|меня|ты|тебя|мы|у меня)\b", lowered):
            return True
        return False

    @staticmethod
    def _looks_like_plain_small_talk(text: str) -> bool:
        lowered = (text or "").strip().lower().replace("ё", "е")
        plain_phrases = {
            "ку",
            "привет",
            "приветик",
            "хай",
            "ало",
            "ау",
            "ок",
            "окей",
            "ага",
            "мм",
            "ясно",
            "пон",
            "спс",
            "?",
            "че как",
            "че как?",
            "как дела",
            "как дела?",
        }
        return lowered in plain_phrases

    def _infer_chat_memory_gender(
        self,
        chat: ApiChat,
        incoming_texts: list[str],
        previous: str,
    ) -> str:
        if chat.is_group:
            return "group/mixed"
        label = chat.chat_label.lower().replace("ё", "е")
        male_names = (
            "влад",
            "владислав",
            "данир",
            "артем",
            "артём",
            "равиль",
            "иван",
            "дима",
            "димон",
            "никита",
            "максим",
            "егор",
            "илья",
            "алексей",
            "саня",
            "фонекс",
            "fonex",
        )
        female_names = (
            "настя",
            "аня",
            "маша",
            "катя",
            "даша",
            "полина",
            "вика",
            "соня",
            "алина",
            "лиза",
        )
        if any(name in label for name in male_names):
            return "male"
        if any(name in label for name in female_names):
            return "female"

        combined = "\n".join(incoming_texts[-40:]).lower().replace("ё", "е")
        male_score = 0
        female_score = 0
        male_patterns = (
            r"\bя\s+(?:был|понял|думал|устал|готов|пошел|пошёл|сказал|сделал|хотел|смог|ожидал|написал|знал)\b",
            r"\bя\s+[а-яa-z]+л\b",
        )
        female_patterns = (
            r"\bя\s+(?:была|поняла|думала|устала|готова|пошла|сказала|сделала|хотела|смогла|ожидала|написала|знала)\b",
            r"\bя\s+[а-яa-z]+ла\b",
        )
        for pattern in male_patterns:
            male_score += len(re.findall(pattern, combined, flags=re.IGNORECASE))
        for pattern in female_patterns:
            female_score += len(re.findall(pattern, combined, flags=re.IGNORECASE))
        if male_score > female_score:
            return "male"
        if female_score > male_score:
            return "female"
        if previous in {"male", "female", "group/mixed"}:
            return previous
        return "unknown"

    def _extract_memory_facts(
        self,
        chat: ApiChat,
        incoming_texts: list[str],
        inferred_gender: str,
    ) -> list[str]:
        facts: list[str] = []
        if not chat.is_group and chat.chat_label and not chat.chat_label.isdigit():
            facts.append(f"chat label/name: {chat.chat_label}")
        if inferred_gender != "unknown":
            facts.append(f"inferred gender: {inferred_gender}")

        for text in incoming_texts[-60:]:
            lowered = text.lower().replace("ё", "е")
            for match in re.finditer(r"(?:меня зовут|зови меня)\s+([A-Za-zА-Яа-яЁё0-9_\- ]{2,32})", text, flags=re.IGNORECASE):
                name = self._clean_memory_value(match.group(1))
                if name:
                    facts.append(f"name mentioned by user: {name}")
            for match in re.finditer(r"(?:мне\s+)?([1-9][0-9])\s*(?:лет|год|года)\b", lowered):
                age = int(match.group(1))
                if 7 <= age <= 80:
                    facts.append(f"age mentioned by user: {age}")
            for match in re.finditer(r"\bмне\s+([1-9][0-9])\b", lowered):
                age = int(match.group(1))
                if 7 <= age <= 80:
                    facts.append(f"age mentioned by user: {age}")
            for match in re.finditer(r"(?:я из|живу в|город)\s+([A-Za-zА-Яа-яЁё\- ]{2,36})", text, flags=re.IGNORECASE):
                city = self._clean_memory_value(match.group(1))
                if city:
                    facts.append(f"city/place mentioned by user: {city}")
            if "девуш" in lowered and ("расстал" in lowered or "расстался" in lowered):
                facts.append("relationship event: user said he broke up with a girlfriend")
            if "огэ" in lowered or "егэ" in lowered:
                facts.append("education context: exams are relevant")
        return facts

    @staticmethod
    def _clean_memory_value(value: str) -> str:
        cleaned = re.split(r"[,.;!?()\n\r]", value.strip(), maxsplit=1)[0].strip(" -:;")
        words = cleaned.split()
        if len(words) > 4:
            cleaned = " ".join(words[:4])
        return cleaned[:80]

    def _build_memory_communication_style(self, chat: ApiChat, incoming_texts: list[str]) -> str:
        if chat.is_group:
            return "group chat; multiple speakers; answer only when the context includes the persona or the persona is already participating"
        recent = [text.strip() for text in incoming_texts[-30:] if text.strip()]
        if not recent:
            return "not enough personal warm messages yet"
        combined = "\n".join(recent).lower().replace("ё", "е")
        avg_len = sum(len(text) for text in recent) / max(1, len(recent))
        traits: list[str] = []
        if avg_len <= 14:
            traits.append("often writes very short messages")
        elif avg_len >= 80:
            traits.append("often writes detailed messages")
        if any(token in combined for token in ("ку", "чё", "че", "пж", "мда", "пон", "щас")):
            traits.append("uses casual slang")
        if any(token in combined for token in ("блять", "нахуй", "пизд", "ебан", "сука", "хуй")):
            traits.append("uses swearing and can be rough")
        if any(token in combined for token in ("люблю", "зай", "мила", "моя", "любим")):
            traits.append("can be affectionate")
        if any(token in combined for token in ("ау", "ответь", "ты тут", "молчишь")):
            traits.append("can be impatient when waiting")
        if any("?" in text for text in recent[-12:]):
            traits.append("often asks direct questions")
        return "; ".join(traits) if traits else "casual neutral texting"

    def _build_memory_relationship_summary(
        self,
        chat: ApiChat,
        inferred_gender: str,
        communication_style: str,
    ) -> str:
        if self._is_forced_reply_chat(chat):
            return (
                "configured partner private chat; keep continuity, warmth, and playful closeness; "
                f"address him as {inferred_gender}; style: {communication_style}"
            )
        if chat.is_group:
            return f"group chat; do not expose private relationship details; style: {communication_style}"
        base = f"private chat; address this person as {inferred_gender}; style: {communication_style}"
        return base

    def _extract_memory_events(self, incoming_texts: list[str]) -> list[str]:
        event_keywords = (
            "расстал",
            "поссор",
            "ссор",
            "боле",
            "забол",
            "экзамен",
            "огэ",
            "егэ",
            "девуш",
            "парн",
            "работ",
            "учеб",
            "школ",
            "переех",
            "люблю",
            "нравишь",
            "ревн",
            "устал",
            "сон",
            "спать",
        )
        events: list[str] = []
        for text in incoming_texts[-40:]:
            lowered = text.lower().replace("ё", "е")
            if any(keyword in lowered for keyword in event_keywords):
                events.append(f"recent user/context event: {text[:140]}")
        return events

    @staticmethod
    def _split_memory_lines(text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
            if line:
                lines.append(line)
        return lines

    def _merge_memory_lines(self, existing_text: str, new_lines: list[str], limit: int) -> str:
        combined = self._split_memory_lines(existing_text) + [
            line.strip()
            for line in new_lines
            if line and line.strip()
        ]
        deduped_reversed: list[str] = []
        seen: set[str] = set()
        for line in reversed(combined):
            clean = re.sub(r"\s+", " ", line).strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_reversed.append(clean[:180])
            if len(deduped_reversed) >= limit:
                break
        return "\n".join(f"- {line}" for line in reversed(deduped_reversed))

    def _build_social_context(self, exclude_chat_id: int) -> str:
        memories = self._store.get_other_chat_memories(
            exclude_chat_id=exclude_chat_id,
            limit=6,
        )
        rows: list[str] = []
        for memory in memories:
            if self._looks_like_service_message(memory.chat_label):
                continue
            details = []
            if memory.relationship_summary:
                details.append(self._compact_memory_text(memory.relationship_summary, 120))
            if memory.recent_events:
                details.append(self._compact_memory_text(memory.recent_events, 150))
            if memory.mood and memory.mood != "neutral":
                details.append(f"mood={memory.mood}")
            if not details:
                continue
            rows.append(f"- {memory.chat_label}: " + " | ".join(details))
            if len(rows) >= 5:
                break
        return "\n".join(rows)

    def _filter_diary_source_messages(
        self,
        source_messages: list[DiarySourceMessage],
    ) -> list[DiarySourceMessage]:
        filtered: list[DiarySourceMessage] = []
        for item in source_messages:
            text = (item.text or "").strip()
            if not text:
                continue
            if self._looks_like_news_message_text(text):
                continue
            if item.direction == "outgoing":
                filtered.append(item)
                continue
            if self._is_tender_message_text(text) or self._looks_like_personal_memory_text(text):
                filtered.append(item)
        return filtered[-80:]

    @staticmethod
    def _compact_memory_text(text: str, max_chars: int) -> str:
        cleaned = re.sub(r"https?://\S+", "[link]", text or "")
        cleaned = re.sub(r"\b\d{4,8}\b", "[number]", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned.replace("- ", " ")).strip()
        return cleaned[:max_chars].rstrip()

    def _build_daily_plan_context(self) -> str:
        now = self._now_moscow()
        plan = self._store.get_daily_plan(now.date().isoformat())
        if plan is None:
            return ""
        current_activity = self._current_plan_activity(plan, now)
        parts = [
            f"Date: {plan.plan_date}",
            f"Now: {now:%Y-%m-%d %H:%M} MSK",
            f"Summary: {plan.summary}",
        ]
        if current_activity:
            parts.append(f"Current planned activity: {current_activity}")
        if plan.schedule:
            parts.append(f"Schedule:\n{plan.schedule}")
        return "\n".join(parts)

    @staticmethod
    def _current_plan_activity(plan: DailyPlan, now: datetime) -> str:
        now_minutes = now.hour * 60 + now.minute
        for raw_line in plan.schedule.splitlines():
            line = raw_line.strip()
            match = re.match(r"(\d{1,2}):(\d{2})\s*(?:-|\u2013|\u2014)\s*(\d{1,2}):(\d{2})\s+(.+)", line)
            if not match:
                continue
            start = int(match.group(1)) * 60 + int(match.group(2))
            end = int(match.group(3)) * 60 + int(match.group(4))
            activity = match.group(5).strip()
            if start <= end:
                if start <= now_minutes < end:
                    return activity
            elif now_minutes >= start or now_minutes < end:
                return activity
        return ""

    @staticmethod
    def _fallback_daily_plan(day_key: str) -> tuple[str, str]:
        return (
            f"The persona has a quiet ordinary day on {day_key}: slow morning, daytime errands/study, evening chats, sleep at night.",
            "\n".join(
                [
                    "07:00-10:00 wakes up slowly, washes up, checks messages",
                    "10:00-14:00 busy with study/chores, replies when free",
                    "14:00-18:00 errands, rest, occasional Telegram",
                    "18:00-22:30 more free, warmer chats, winding down",
                    "23:00-07:00 sleep/offline",
                ]
            ),
        )

    def _build_chat_snapshot_image_input(
        self,
        *,
        chat_label: str,
        history: list,
        latest_incoming_text: str,
    ) -> ImageInput | None:
        if not self._settings.vision_enabled:
            return None
        try:
            width = 920
            bubble_width = 700
            padding = 28
            line_height = 28
            rows: list[tuple[str, str]] = []
            for item in history[-8:]:
                role = "outgoing" if item.role == "assistant" else "incoming"
                rows.append((role, item.content))
            if not rows or rows[-1][1] != latest_incoming_text:
                rows.append(("incoming", latest_incoming_text))

            font = ImageFont.load_default()
            prepared: list[tuple[str, list[str]]] = []
            total_lines = 0
            for role, content in rows:
                text = (content or "").replace("\n", " ").strip() or "..."
                wrapped = self._wrap_text_for_snapshot(text, 44)
                prepared.append((role, wrapped))
                total_lines += len(wrapped) + 1

            height = max(720, padding * 3 + total_lines * line_height + 120)
            image = Image.new("RGB", (width, height), (233, 240, 248))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 86), fill=(248, 250, 252))
            draw.text((padding, 28), f"Telegram: {chat_label}", fill=(35, 42, 52), font=font)

            y = 110
            for role, wrapped in prepared:
                lines_count = len(wrapped)
                bubble_h = 24 + lines_count * line_height
                if role == "outgoing":
                    x1 = width - padding - bubble_width
                    x2 = width - padding
                    fill = (212, 248, 214)
                else:
                    x1 = padding
                    x2 = padding + bubble_width
                    fill = (255, 255, 255)
                draw.rounded_rectangle((x1, y, x2, y + bubble_h), radius=18, fill=fill, outline=(198, 205, 214))
                text_y = y + 12
                for line in wrapped:
                    draw.text((x1 + 18, text_y), line, fill=(30, 36, 44), font=font)
                    text_y += line_height
                y += bubble_h + 18

            draw.rounded_rectangle((padding, height - 88, width - padding, height - 28), radius=18, fill=(255, 255, 255), outline=(198, 205, 214))
            draw.text((padding + 18, height - 69), "Сообщение...", fill=(120, 130, 145), font=font)

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return ImageInput(
                mime_type="image/png",
                base64_data=base64.b64encode(buffer.getvalue()).decode("ascii"),
            )
        except Exception:
            return None

    @staticmethod
    def _wrap_text_for_snapshot(text: str, max_chars: int) -> list[str]:
        words = text.split()
        if not words:
            return ["..."]
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines[:8]

    def _gallery_files(self) -> list[Path]:
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(self._settings.gallery_dir.glob(pattern)))
        return files

    def _avatar_files(self) -> list[Path]:
        avatar_dir = self._settings.gallery_dir / "avatars"
        target_dir = avatar_dir if avatar_dir.exists() else self._settings.gallery_dir
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(target_dir.glob(pattern)))
        return files

    def _load_gallery_reference_image_input(self, latest_incoming_text: str) -> ImageInput | None:
        lowered = latest_incoming_text.lower().replace("ё", "е")
        if not any(token in lowered for token in ("как выглядишь", "фото", "селфи", "покажи себя", "ты красивая", "как ты выглядишь")):
            return None
        gallery_files = self._gallery_files()
        if not gallery_files:
            return None
        path = gallery_files[0]
        mime_type = "image/jpeg"
        if path.suffix.lower() == ".png":
            mime_type = "image/png"
        elif path.suffix.lower() == ".webp":
            mime_type = "image/webp"
        return ImageInput(
            mime_type=mime_type,
            base64_data=base64.b64encode(path.read_bytes()).decode("ascii"),
        )

    def _pick_gallery_photo_for_send(
        self,
        chat: ApiChat,
        latest_incoming_text: str,
        rating: int,
    ) -> str | None:
        if chat.is_group:
            return None
        lowered = latest_incoming_text.lower().replace("ё", "е")
        if not any(token in lowered for token in ("скинь фото", "покажи фото", "покажи себя", "селфи", "как выглядишь")):
            return None
        if not self._is_forced_reply_chat(chat) and rating < 15:
            return None
        gallery_files = self._gallery_files()
        if not gallery_files:
            return None
        return str(random.choice(gallery_files))

    @staticmethod
    def _message_may_need_runtime_action(text: str) -> bool:
        lowered = text.lower().replace("ё", "е")
        return any(
            token in lowered
            for token in (
                "аватар",
                "аватарка",
                "ава",
                "аву",
                "поставь аву",
                "смени аву",
                "смени аватар",
                "поменяй аву",
                "поменяй аватар",
                "фото профиля",
            )
        )

    async def _decide_runtime_action(
        self,
        chat: ApiChat,
        latest_incoming_text: str,
        history,
    ) -> RuntimeActionRequest:
        if not self._message_may_need_runtime_action(latest_incoming_text):
            return RuntimeActionRequest(
                should_run=False,
                action_name="none",
                action_arg="",
                reason="no action cue",
                raw_response={},
            )
        avatar_files = self._avatar_files()
        if not avatar_files:
            return RuntimeActionRequest(
                should_run=False,
                action_name="none",
                action_arg="",
                reason="no avatar files",
                raw_response={},
            )
        return await self._responder.decide_runtime_action(
            chat_label=chat.chat_label,
            latest_incoming_text=latest_incoming_text,
            history=history,
            available_avatar_files=[path.name for path in avatar_files],
        )

    async def _execute_runtime_action(
        self,
        chat: ApiChat,
        action_request: RuntimeActionRequest,
    ) -> None:
        assert self._client is not None
        if not action_request.should_run:
            return
        if action_request.action_name == "set_avatar":
            avatar_files = self._avatar_files()
            if not avatar_files:
                LOGGER.info("Runtime action skipped in %s: no avatar files", chat.chat_label)
                return
            selected_path: Path | None = None
            if action_request.action_arg and action_request.action_arg.lower() != "random":
                for path in avatar_files:
                    if path.name.lower() == action_request.action_arg.lower():
                        selected_path = path
                        break
            if selected_path is None:
                selected_path = random.choice(avatar_files)
            uploaded = await self._client.upload_file(str(selected_path))
            await self._client(tl_functions.photos.UploadProfilePhotoRequest(file=uploaded))
            LOGGER.info(
                "Runtime action executed in %s: set_avatar -> %s%s",
                chat.chat_label,
                selected_path.name,
                f" | {action_request.reason}" if action_request.reason else "",
            )

    async def _chat_from_event(self, event) -> ApiChat | None:
        entity = await event.get_chat()
        is_direct = isinstance(entity, User)
        is_group = False

        if isinstance(entity, User):
            if getattr(entity, "bot", False):
                return None
            chat_label = self._format_user_label(entity)
        elif isinstance(entity, Chat):
            chat_label = (getattr(entity, "title", None) or "").strip()
            is_group = True
        elif isinstance(entity, Channel):
            if getattr(entity, "broadcast", False):
                return None
            chat_label = (getattr(entity, "title", None) or "").strip()
            is_group = True
        else:
            return None
        if not chat_label:
            return None
        chat_id = int(getattr(event, "chat_id", 0) or getattr(entity, "id", 0))
        if not chat_id:
            return None
        return ApiChat(
            chat_id=chat_id,
            chat_label=chat_label,
            entity=entity,
            is_direct=is_direct,
            is_group=is_group,
        )

    async def _iter_candidate_chats(self) -> list[ApiChat]:
        assert self._client is not None
        dialogs: list[ApiChat] = []
        async for dialog in self._client.iter_dialogs(limit=40):
            chat = self._dialog_to_chat(dialog)
            if chat is None:
                continue
            if not self._is_allowed_chat(chat.chat_label):
                continue
            dialogs.append(chat)
        return dialogs

    async def _iter_unread_message_dialogs(self) -> list[ApiChat]:
        assert self._client is not None
        dialogs: list[ApiChat] = []
        async for dialog in self._client.iter_dialogs(limit=80):
            if int(getattr(dialog, "unread_count", 0) or 0) <= 0:
                continue
            chat = self._dialog_to_message_chat(dialog)
            if chat is None:
                continue
            if not self._is_allowed_chat(chat.chat_label):
                continue
            dialogs.append(chat)
        return dialogs

    def _dialog_to_chat(self, dialog: Dialog) -> ApiChat | None:
        entity = dialog.entity
        if not isinstance(entity, User):
            return None
        if getattr(entity, "bot", False):
            return None
        chat_label = (dialog.name or "").strip() or self._format_user_label(entity)
        if not chat_label:
            return None
        if self._is_service_or_system_chat(entity, chat_label):
            return None
        return ApiChat(
            chat_id=dialog.id,
            chat_label=chat_label,
            entity=entity,
            is_direct=True,
            is_group=False,
        )

    def _dialog_to_message_chat(self, dialog: Dialog) -> ApiChat | None:
        entity = dialog.entity
        if isinstance(entity, User):
            if getattr(entity, "bot", False):
                return None
            chat_label = (dialog.name or "").strip() or self._format_user_label(entity)
            if not chat_label or self._is_service_or_system_chat(entity, chat_label):
                return None
            return ApiChat(
                chat_id=dialog.id,
                chat_label=chat_label,
                entity=entity,
                is_direct=True,
                is_group=False,
            )
        if isinstance(entity, Chat):
            chat_label = (dialog.name or "").strip() or (getattr(entity, "title", None) or "").strip()
            if not chat_label:
                return None
            return ApiChat(
                chat_id=dialog.id,
                chat_label=chat_label,
                entity=entity,
                is_direct=False,
                is_group=True,
            )
        if isinstance(entity, Channel) and not getattr(entity, "broadcast", False):
            chat_label = (dialog.name or "").strip() or (getattr(entity, "title", None) or "").strip()
            if not chat_label:
                return None
            return ApiChat(
                chat_id=dialog.id,
                chat_label=chat_label,
                entity=entity,
                is_direct=False,
                is_group=True,
            )
        return None

    @staticmethod
    def _is_service_or_system_chat(entity: User, chat_label: str) -> bool:
        lowered_label = chat_label.strip().lower()
        username = (getattr(entity, "username", None) or "").strip().lower()
        first_name = (getattr(entity, "first_name", None) or "").strip().lower()
        if lowered_label in {"telegram", "telegram support", "telegram notifications", "login code"}:
            return True
        if lowered_label.startswith("telegram "):
            return True
        if username in {"telegram", "smstelegram"}:
            return True
        if first_name == "telegram":
            return True
        if getattr(entity, "support", False):
            return True
        if getattr(entity, "verified", False) and username == "telegram":
            return True
        return False

    async def _iter_channel_dialogs(self) -> list[ApiChannel]:
        assert self._client is not None
        channels: list[ApiChannel] = []
        async for dialog in self._client.iter_dialogs(limit=80):
            entity = dialog.entity
            if not isinstance(entity, Channel):
                continue
            if not getattr(entity, "broadcast", False):
                continue
            chat_label = (dialog.name or "").strip() or str(dialog.id)
            channels.append(
                ApiChannel(
                    chat_id=dialog.id,
                    chat_label=chat_label,
                    entity=entity,
                    unread_count=int(getattr(dialog, "unread_count", 0) or 0),
                )
            )
        return channels

    def _build_client_kwargs(self) -> dict:
        kwargs = {
            "timeout": 15,
            "request_retries": 8,
            "connection_retries": 8,
            "retry_delay": 2,
            "auto_reconnect": True,
        }
        proxy_type = self._settings.telegram_proxy_type
        if not proxy_type:
            return kwargs

        proxy_host = self._settings.telegram_proxy_host
        proxy_port = self._settings.telegram_proxy_port
        if not proxy_host or not proxy_port:
            raise RuntimeError(
                "Telegram proxy is enabled but TELEGRAM_PROXY_HOST or TELEGRAM_PROXY_PORT is missing"
            )

        if proxy_type == "mtproto":
            secret = self._settings.telegram_proxy_secret
            if not secret:
                raise RuntimeError("TELEGRAM_PROXY_SECRET is required for TELEGRAM_PROXY_TYPE=mtproto")
            kwargs["connection"] = (
                ConnectionTcpMTProxyRandomizedIntermediate
                if secret.lower().startswith("dd")
                else ConnectionTcpMTProxyIntermediate
            )
            kwargs["proxy"] = (proxy_host, proxy_port, secret)
            return kwargs

        if proxy_type in {"socks5", "socks4", "http"}:
            kwargs["proxy"] = (
                proxy_type,
                proxy_host,
                proxy_port,
                True,
                self._settings.telegram_proxy_username,
                self._settings.telegram_proxy_password,
            )
            return kwargs

        raise RuntimeError(
            "TELEGRAM_PROXY_TYPE must be one of: mtproto, socks5, socks4, http"
        )

    async def _sync_and_maybe_reply(
        self,
        chat: ApiChat,
        revision: int | None = None,
        baseline_id: int | None = None,
    ) -> None:
        assert self._client is not None
        stored_latest_id = self._store.get_last_incoming_message_id(chat.chat_id)
        previous_latest_id = (
            baseline_id
            if baseline_id is not None
            else self._get_previous_incoming_checkpoint(chat.chat_id)
        )
        is_initial_sync = stored_latest_id is None and chat.chat_id not in self._last_handled_incoming_id_by_chat
        messages = await self._client.get_messages(chat.entity, limit=max(self._settings.chat_history_limit, 20))
        latest_incoming_id = previous_latest_id
        latest_incoming_text = ""
        latest_message_is_outgoing = False
        latest_incoming_images: list[ImageInput] = []
        latest_incoming_at = ""
        latest_reply_to_msg_id: int | None = None
        incoming_burst: list[tuple[Any, str]] = []

        for message in reversed(messages):
            if not getattr(message, "id", None):
                continue
            text = self._format_message_for_storage(message)
            if not text:
                continue
            direction = "outgoing" if message.out else "incoming"
            sender_name = None if message.out else chat.chat_label
            self._store.add_message(
                chat_id=chat.chat_id,
                message_id=message.id,
                direction=direction,
                text=text,
                chat_label=chat.chat_label,
                sender_name=sender_name,
                created_at=self._format_moscow_timestamp(message.date),
            )
        self._refresh_chat_memory(chat)

        for message in messages:
            text = self._format_message_for_storage(message)
            if not text:
                continue
            latest_message_is_outgoing = bool(message.out)
            if latest_message_is_outgoing:
                break
            message_id = int(message.id)
            if message_id <= previous_latest_id:
                break
            incoming_burst.append((message, text))
            if len(incoming_burst) >= 5:
                break

        if not incoming_burst:
            return

        latest_incoming_id = int(incoming_burst[0][0].id)
        latest_incoming_at = self._format_moscow_timestamp(incoming_burst[0][0].date)
        latest_reply_to = getattr(incoming_burst[0][0], "reply_to", None)
        latest_reply_to_msg_id = getattr(latest_reply_to, "reply_to_msg_id", None)
        latest_message_is_outgoing = False
        latest_incoming_text = self._build_incoming_burst_text(
            [text for _, text in reversed(incoming_burst)]
        )
        for message, text in reversed(incoming_burst):
            if self._is_photo_placeholder_text(text):
                continue
            extracted = await self._extract_image_inputs(message)
            if extracted:
                latest_incoming_images.extend(extracted)
                latest_incoming_images = latest_incoming_images[: self._settings.max_images_per_message]

        if is_initial_sync:
            self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
            LOGGER.info("Initial sync completed for %s", chat.chat_label)
            return

        if latest_incoming_images and self._is_photo_placeholder_text(latest_incoming_text):
            self._cache_pending_image_context(chat.chat_id, latest_incoming_images)
            self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
            LOGGER.info("Cached photo context in %s", chat.chat_label)
            return

        if self._is_sleep_time_moscow():
            LOGGER.info(
                "Sleep mode active, left unread in %s: %s",
                chat.chat_label,
                latest_incoming_text,
            )
            return
        await self._mark_chat_read(chat, latest_incoming_id)

        group_triggered = False
        if chat.is_group:
            group_triggered = await self._should_reply_in_group(
                chat,
                latest_incoming_text,
                latest_reply_to_msg_id,
            )
            if not group_triggered:
                self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
                LOGGER.info(
                    "Ignored group message in %s without relevant context: %s",
                    chat.chat_label,
                    latest_incoming_text,
                )
                return
            latest_incoming_text = self._strip_group_trigger_prefix(latest_incoming_text)

        if self._settings.trigger_prefix:
            if not latest_incoming_text.startswith(self._settings.trigger_prefix):
                LOGGER.info(
                    "Latest message in %s ignored because prefix is missing",
                    chat.chat_label,
                )
                self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
                return
            stripped = latest_incoming_text[len(self._settings.trigger_prefix) :].strip()
            if stripped:
                latest_incoming_text = stripped

        if not latest_incoming_images:
            latest_incoming_images = self._take_pending_image_context(chat.chat_id, latest_incoming_text)
            if latest_incoming_images:
                LOGGER.info("Attached cached photo context in %s", chat.chat_label)
        history = self._store.get_recent_messages(chat.chat_id, max(24, self._settings.chat_history_limit))
        if self._should_attach_chat_snapshot(latest_incoming_text, len(incoming_burst)):
            snapshot_input = self._build_chat_snapshot_image_input(
                chat_label=chat.chat_label,
                history=history,
                latest_incoming_text=latest_incoming_text,
            )
            if snapshot_input is not None:
                latest_incoming_images = [snapshot_input, *latest_incoming_images][: self._settings.max_images_per_message + 1]

        gallery_reference = self._load_gallery_reference_image_input(latest_incoming_text)
        if gallery_reference is not None:
            latest_incoming_images = [*latest_incoming_images, gallery_reference][: self._settings.max_images_per_message + 2]

        diary_entries = self._select_diary_context(chat.chat_id)
        force_reply = self._is_forced_reply_chat(chat) or group_triggered
        profile = self._store.get_user_profile(chat.chat_id, chat.chat_label)
        gallery_photo_path = self._pick_gallery_photo_for_send(chat, latest_incoming_text, profile.rating)
        weather_context = await self._responder.get_moscow_weather_context(latest_incoming_text)
        action_request = await self._decide_runtime_action(chat, latest_incoming_text, history)
        reply_messages, sticker_decision = await self._generate_reply_bundle(
            chat=chat,
            latest_incoming_text=latest_incoming_text,
            history=history,
            diary_entries=diary_entries,
            image_inputs=latest_incoming_images,
            force_reply=force_reply,
            source_message_id=latest_incoming_id,
            latest_incoming_at=latest_incoming_at,
            weather_context=weather_context,
        )
        if revision is not None and self._reply_revision_by_chat.get(chat.chat_id) != revision:
            LOGGER.info("Dropped stale reply result in %s", chat.chat_label)
            return
        if not reply_messages and not sticker_decision.should_send and not gallery_photo_path and not action_request.should_run:
            self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
            return

        if action_request.should_run:
            await self._execute_runtime_action(chat, action_request)
        self._last_handled_incoming_id_by_chat[chat.chat_id] = latest_incoming_id
        await self._send_reply(
            chat,
            reply_messages,
            latest_incoming_text,
            sticker_decision,
            source_message_id=latest_incoming_id,
            gallery_photo_path=gallery_photo_path,
            revision=revision,
        )

    async def _mark_chat_read(self, chat: ApiChat, latest_incoming_id: int) -> None:
        assert self._client is not None
        previous_ack_id = self._last_read_ack_id_by_chat.get(chat.chat_id, 0)
        if latest_incoming_id <= previous_ack_id:
            return
        await self._client.send_read_acknowledge(
            chat.entity,
            max_id=latest_incoming_id,
            clear_mentions=True,
            clear_reactions=True,
        )
        self._last_read_ack_id_by_chat[chat.chat_id] = latest_incoming_id

    async def _send_reply(
        self,
        chat: ApiChat,
        reply_messages: list[str],
        latest_incoming_text: str,
        sticker_decision: StickerDecision,
        source_message_id: int | None = None,
        gallery_photo_path: str | None = None,
        revision: int | None = None,
    ) -> None:
        assert self._client is not None
        if not self._settings.auto_send:
            LOGGER.info("Latest incoming in %s: %s", chat.chat_label, latest_incoming_text)
            if reply_messages:
                LOGGER.info("Draft for %s: %s", chat.chat_label, reply_messages[0])
            if gallery_photo_path:
                LOGGER.info("Gallery photo draft for %s: %s", chat.chat_label, Path(gallery_photo_path).name)
            if sticker_decision.should_send:
                LOGGER.info(
                    "Sticker draft for %s: id=%s | mode=%s%s",
                    chat.chat_label,
                    sticker_decision.candidate_id,
                    sticker_decision.mode,
                    f" | {sticker_decision.reason}" if sticker_decision.reason else "",
                )
            return

        if revision is not None and self._reply_revision_by_chat.get(chat.chat_id) != revision:
            LOGGER.info("Skipped stale send in %s", chat.chat_label)
            return

        reply_to = None
        if source_message_id is not None:
            current_last_incoming = self._last_handled_incoming_id_by_chat.get(chat.chat_id, 0)
            if current_last_incoming > source_message_id:
                reply_to = source_message_id
        if reply_messages:
            async with self._client.action(chat.entity, "typing"):
                if self._settings.fast_reply_mode:
                    await asyncio.sleep(random.uniform(0.02, 0.06))
                else:
                    await asyncio.sleep(random.uniform(0.12, 0.35))
        if reply_messages and sticker_decision.mode != "only":
            if revision is not None and self._reply_revision_by_chat.get(chat.chat_id) != revision:
                LOGGER.info("Skipped stale text send in %s", chat.chat_label)
                return
            reply_text = reply_messages[0]
            sent = await self._client.send_message(chat.entity, reply_text, reply_to=reply_to)
            self._last_interactive_event_at = datetime.now().astimezone()
            self._store.add_message(
                chat_id=chat.chat_id,
                message_id=sent.id,
                direction="outgoing",
                text=reply_text,
                chat_label=chat.chat_label,
            )
        if sticker_decision.should_send:
            if revision is not None and self._reply_revision_by_chat.get(chat.chat_id) != revision:
                LOGGER.info("Skipped stale sticker send in %s", chat.chat_label)
                return
            sent_sticker = await self._send_sticker(chat, sticker_decision)
            if sent_sticker:
                LOGGER.info("Sticker sent to %s", chat.chat_label)
        if gallery_photo_path:
            if revision is not None and self._reply_revision_by_chat.get(chat.chat_id) != revision:
                LOGGER.info("Skipped stale gallery send in %s", chat.chat_label)
                return
            with suppress(Exception):
                sent_photo = await self._client.send_file(chat.entity, gallery_photo_path, reply_to=reply_to)
                self._store.add_message(
                    chat_id=chat.chat_id,
                    message_id=sent_photo.id,
                    direction="outgoing",
                    text=f"[Photo] {Path(gallery_photo_path).name}",
                    chat_label=chat.chat_label,
                )
                LOGGER.info("Gallery photo sent to %s", chat.chat_label)
        LOGGER.info("Reply sent to %s", chat.chat_label)
        await self._set_online_presence(False, "after reply")

    async def _generate_reply_bundle(
        self,
        *,
        chat: ApiChat,
        latest_incoming_text: str,
        history,
        diary_entries: list[DiaryEntry],
        image_inputs: list[ImageInput],
        force_reply: bool,
        source_message_id: int | None,
        latest_incoming_at: str | None,
        weather_context: str | None,
    ) -> tuple[list[str], StickerDecision]:
        assert self._client is not None
        profile = self._store.get_user_profile(chat.chat_id, chat.chat_label)
        chat_memory = self._store.get_chat_memory(chat.chat_id, chat.chat_label)
        if self._is_forced_reply_chat(chat):
            chat_memory = self._store.update_chat_emotions(
                chat_id=chat.chat_id,
                chat_label=chat.chat_label,
                affection_delta=max(0, 82 - chat_memory.affection),
                trust_delta=max(0, 72 - chat_memory.trust),
                loyalty_delta=max(0, 92 - chat_memory.loyalty),
                attachment_delta=max(0, 84 - chat_memory.attachment),
                jealousy_delta=0,
                irritation_delta=0,
            )
        if self._is_forced_reply_chat(chat) and profile.rating < 100:
            profile = self._store.set_user_profile(
                chat_id=chat.chat_id,
                chat_label=chat.chat_label,
                rating=100,
                notes="partner / highest trust",
            )
        reply = await self._responder.generate_text_reply(
            chat_label=chat.chat_label,
            is_group=chat.is_group,
            latest_incoming_text=latest_incoming_text,
            history=history,
            diary_entries=diary_entries,
            image_inputs=image_inputs,
            force_reply=force_reply,
            user_rating=profile.rating,
            user_notes=profile.notes,
            chat_memory=chat_memory,
            social_context=self._build_social_context(chat.chat_id),
            daily_plan_context=self._build_daily_plan_context(),
            latest_incoming_at=latest_incoming_at,
            weather_context=weather_context,
        )
        reply_messages = [item.strip() for item in reply.messages if item.strip()][:1]
        if not reply_messages:
            sticker_decision = StickerDecision(
                should_send=False,
                candidate_id=None,
                mode="none",
                reason="no text reply",
                raw_response={},
            )
        elif self._settings.fast_reply_mode and self._settings.fast_reply_skip_stickers:
            sticker_decision = StickerDecision(
                should_send=False,
                candidate_id=None,
                mode="none",
                reason="fast reply mode",
                raw_response={},
            )
        else:
            sticker_decision = await self._decide_sticker(
                chat=chat,
                latest_incoming_text=latest_incoming_text,
                planned_reply_text="\n".join(reply_messages).strip(),
                history=history,
                diary_entries=diary_entries,
            )
        return reply_messages if (reply.should_reply or force_reply) else [], sticker_decision

    def _is_forced_reply_chat(self, chat: ApiChat) -> bool:
        if chat.is_group:
            return False
        if chat.chat_id in self._settings.partner_chat_ids:
            return True
        lowered = chat.chat_label.lower()
        return any(token and token in lowered for token in self._settings.partner_chat_names)

    async def _resolve_sender_name(self, event, chat: ApiChat) -> str | None:
        if not chat.is_group:
            return chat.chat_label
        sender = await event.get_sender()
        if isinstance(sender, User):
            return self._format_user_label(sender)
        return (getattr(sender, "title", None) or "").strip() or None

    def _matches_group_reply_trigger(self, text: str) -> bool:
        normalized = self._normalize_trigger_text(text)
        if not normalized:
            return False

        digits_only = re.sub(r"\D+", "", text)
        for trigger in self._settings.group_reply_triggers:
            lowered = trigger.strip().lower()
            if not lowered:
                continue
            if lowered.startswith("+") or lowered.isdigit():
                trigger_digits = re.sub(r"\D+", "", lowered)
                if trigger_digits and trigger_digits in digits_only:
                    return True
                continue
            if lowered.startswith("@"):
                if lowered in normalized:
                    return True
                continue
            if re.search(rf"(?<![\w@]){re.escape(lowered)}(?![\w@])", normalized):
                return True
        return False

    def _strip_group_trigger_prefix(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text

        patterns = [
            rf"^\s*{re.escape(trigger)}[\s,:;\-]*"
            for trigger in self._settings.group_reply_triggers
            if trigger and not trigger.startswith("+") and not trigger.isdigit()
        ]
        patterns.extend(
            r"^\s*\+?[\d\-\s\(\)]{7,}[\s,:;\-]*"
            for trigger in self._settings.group_reply_triggers
            if trigger.startswith("+") or trigger.isdigit()
        )
        updated = stripped
        for pattern in patterns:
            candidate = re.sub(pattern, "", updated, count=1, flags=re.IGNORECASE)
            if candidate != updated:
                updated = candidate.strip()
        return updated or stripped

    @staticmethod
    def _normalize_trigger_text(text: str) -> str:
        return text.strip().lower().replace("ё", "е")

    def _is_forced_reply_chat(self, chat: ApiChat) -> bool:
        if chat.is_group:
            return False
        if chat.chat_id in self._settings.partner_chat_ids:
            return True
        lowered = chat.chat_label.lower()
        return any(token and token in lowered for token in self._settings.partner_chat_names)

    async def _resolve_sender_name(self, event, chat: ApiChat) -> str | None:
        if not chat.is_group:
            return chat.chat_label
        sender = await event.get_sender()
        if isinstance(sender, User):
            return self._format_user_label(sender)
        return (getattr(sender, "title", None) or "").strip() or None

    def _matches_group_reply_trigger(self, text: str) -> bool:
        normalized = self._normalize_trigger_text(text)
        if not normalized:
            return False

        digits_only = re.sub(r"\D+", "", text)
        for trigger in self._settings.group_reply_triggers:
            lowered = trigger.strip().lower()
            if not lowered:
                continue
            if lowered.startswith("+") or lowered.isdigit():
                trigger_digits = re.sub(r"\D+", "", lowered)
                if trigger_digits and trigger_digits in digits_only:
                    return True
                continue
            if lowered.startswith("@"):
                if lowered in normalized:
                    return True
                continue
            if re.search(rf"(?<![\w@]){re.escape(lowered)}(?![\w@])", normalized):
                return True
        return False

    def _strip_group_trigger_prefix(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text

        patterns = [
            rf"^\s*{re.escape(trigger)}[\s,:;\-]*"
            for trigger in self._settings.group_reply_triggers
            if trigger and not trigger.startswith("+") and not trigger.isdigit()
        ]
        patterns.extend(
            r"^\s*\+?[\d\-\s\(\)]{7,}[\s,:;\-]*"
            for trigger in self._settings.group_reply_triggers
            if trigger.startswith("+") or trigger.isdigit()
        )
        updated = stripped
        for pattern in patterns:
            candidate = re.sub(pattern, "", updated, count=1, flags=re.IGNORECASE)
            if candidate != updated:
                updated = candidate.strip()
        return updated or stripped

    @staticmethod
    def _normalize_trigger_text(text: str) -> str:
        return text.strip().lower().replace("\u0451", "\u0435")

    def _select_diary_context(self, chat_id: int) -> list[DiaryEntry]:
        if not self._settings.diary_enabled:
            return []
        history = self._store.get_recent_messages(chat_id, self._settings.chat_history_limit)
        query_text = "\n".join(item.content for item in history[-6:])
        if not query_text.strip():
            return []
        return self._store.get_relevant_diary_entries(
            query_text=query_text,
            limit=self._settings.diary_lookup_limit,
            lookback_days=self._settings.diary_lookback_days,
        )

    async def _extract_image_inputs(self, message) -> list[ImageInput]:
        assert self._client is not None
        image_inputs: list[ImageInput] = []
        if not self._settings.vision_enabled:
            return image_inputs
        if len(image_inputs) >= self._settings.max_images_per_message:
            return image_inputs

        if isinstance(getattr(message, "media", None), MessageMediaPhoto) or getattr(message, "photo", None):
            raw = await self._client.download_media(message, file=bytes)
            if isinstance(raw, (bytes, bytearray)) and raw:
                image_inputs.append(
                    ImageInput(
                        mime_type="image/jpeg",
                        base64_data=base64.b64encode(bytes(raw)).decode("ascii"),
                    )
                )
            return image_inputs[: self._settings.max_images_per_message]

        document = getattr(message, "document", None)
        if document is None or not getattr(document, "mime_type", "").startswith("image/"):
            return image_inputs

        raw = await self._client.download_media(message, file=bytes)
        if isinstance(raw, (bytes, bytearray)) and raw:
            image_inputs.append(
                ImageInput(
                    mime_type=document.mime_type or "image/jpeg",
                    base64_data=base64.b64encode(bytes(raw)).decode("ascii"),
                )
            )
        return image_inputs[: self._settings.max_images_per_message]

    def _format_message_for_storage(self, message) -> str:
        text = (getattr(message, "message", None) or "").strip()
        if text:
            return text
        if getattr(message, "sticker", False):
            emoji = self._extract_sticker_emoji(getattr(message, "document", None))
            return f"[Sticker] {emoji}".strip()
        if isinstance(getattr(message, "media", None), MessageMediaPhoto) or getattr(message, "photo", None):
            return "[Photo]"
        document = getattr(message, "document", None)
        if document is not None and getattr(document, "mime_type", "").startswith("image/"):
            filename = ""
            for attr in getattr(document, "attributes", []) or []:
                if isinstance(attr, DocumentAttributeFilename):
                    filename = (attr.file_name or "").strip()
                    break
            return f"[Image] {filename}".strip()
        return ""

    async def _diary_loop(self) -> None:
        while True:
            try:
                if self._should_prioritize_reply_latency():
                    await asyncio.sleep(min(20, max(5, self._settings.poll_interval_seconds * 4)))
                    continue
                await self._refresh_diary_if_needed()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Diary refresh failed: %s", exc)
            await asyncio.sleep(self._settings.diary_check_interval_seconds)

    async def _proactive_loop(self) -> None:
        while True:
            try:
                if self._should_prioritize_reply_latency():
                    await asyncio.sleep(min(20, max(5, self._settings.poll_interval_seconds * 4)))
                    continue
                await self._maybe_send_proactive_message()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Proactive loop failed: %s", exc)
            await asyncio.sleep(self._settings.proactive_check_interval_seconds)

    async def _online_presence_loop(self) -> None:
        while True:
            try:
                if self._should_prioritize_reply_latency():
                    await asyncio.sleep(min(20, max(5, self._settings.poll_interval_seconds * 4)))
                    continue
                await self._maybe_pulse_online_presence()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Online presence loop failed: %s", exc)
            await asyncio.sleep(self._settings.online_presence_check_interval_seconds)

    async def _sync_channels(self) -> None:
        if not self._settings.channels_auto_read:
            return
        if self._should_prioritize_reply_latency():
            return
        if self._is_sleep_time_moscow():
            return
        now = datetime.now().astimezone()
        if self._last_channel_sync_at is not None:
            if (now - self._last_channel_sync_at) < timedelta(seconds=max(45, int(self._settings.poll_interval_seconds * 10))):
                return
        self._last_channel_sync_at = now
        for channel in await self._iter_channel_dialogs():
            if channel.unread_count <= 0 and self._store.get_last_incoming_message_id(channel.chat_id) is not None:
                continue
            try:
                await self._sync_channel_and_mark_read(channel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Channel sync failed for %s: %s", channel.chat_label, exc)

    async def _sync_channel_and_mark_read(self, channel: ApiChannel) -> None:
        assert self._client is not None
        stored_latest_id = self._store.get_last_incoming_message_id(channel.chat_id) or 0
        messages = await self._client.get_messages(
            channel.entity,
            limit=max(self._settings.channel_history_limit, 5),
        )
        latest_channel_id = stored_latest_id

        for message in reversed(messages):
            if not getattr(message, "id", None):
                continue
            text = self._format_message_for_storage(message)
            if not text:
                continue
            latest_channel_id = max(latest_channel_id, int(message.id))
            self._store.add_message(
                chat_id=channel.chat_id,
                message_id=message.id,
                direction="incoming",
                text=text,
                chat_label=channel.chat_label,
                sender_name=channel.chat_label,
                created_at=self._format_moscow_timestamp(message.date),
            )

        if latest_channel_id <= stored_latest_id and channel.unread_count <= 0:
            return

        previous_ack_id = self._last_read_ack_id_by_channel.get(channel.chat_id, 0)

        if latest_channel_id > 0:
            await self._client.send_read_acknowledge(
                channel.entity,
                max_id=latest_channel_id,
                clear_mentions=True,
                clear_reactions=True,
            )
        else:
            await self._client.send_read_acknowledge(
                channel.entity,
                clear_mentions=True,
                clear_reactions=True,
            )
        if latest_channel_id > 0:
            self._last_read_ack_id_by_channel[channel.chat_id] = latest_channel_id

        if latest_channel_id > max(stored_latest_id, previous_ack_id) or channel.unread_count > 0:
            LOGGER.info("Channel read and synced: %s", channel.chat_label)

    async def _maybe_process_wakeup_backlog(self) -> None:
        if self._is_sleep_time_moscow():
            return
        now_msk = self._now_moscow()
        if now_msk.hour < 7:
            return
        day_key = now_msk.date().isoformat()
        if self._last_sleep_catchup_day == day_key:
            return
        chats = await self._iter_unread_message_dialogs()
        if not chats:
            self._last_sleep_catchup_day = day_key
            return
        LOGGER.info("Wakeup backlog processing started: %s chats", len(chats))
        for chat in chats:
            try:
                await self._sync_and_maybe_reply(
                    chat,
                    baseline_id=self._get_unread_backlog_checkpoint(chat.chat_id),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Wakeup backlog failed for %s: %s", chat.chat_label, exc)
            await asyncio.sleep(0.2)
        self._last_sleep_catchup_day = day_key
        LOGGER.info("Wakeup backlog processing finished")

    async def _maybe_pulse_online_presence(self) -> None:
        assert self._client is not None
        if self._is_sleep_time_moscow():
            return
        now = datetime.now().astimezone()
        if self._last_interactive_event_at is not None:
            if now - self._last_interactive_event_at < timedelta(minutes=5):
                return
        if self._last_online_pulse_at is not None:
            since_last = now - self._last_online_pulse_at.astimezone()
            if since_last < timedelta(minutes=self._settings.online_presence_cooldown_minutes):
                return

        recent_rows: list[str] = []
        for chat in await self._iter_candidate_chats():
            last_message = self._store.get_last_message(chat.chat_id)
            if last_message is None:
                continue
            recent_rows.append(
                f"- {chat.chat_label}: last={last_message.created_at}, direction={last_message.direction}, text={last_message.text[:80]}"
            )
            if len(recent_rows) >= 6:
                break

        local_time_text = now.strftime("%Y-%m-%d %H:%M (%A)")
        if 1 <= now.hour <= 6 and not recent_rows:
            return

        decision = await self._responder.decide_online_presence(
            local_time_text=local_time_text,
            recent_activity_text="\n".join(recent_rows) if recent_rows else "- no recent chats",
        )
        if not decision.should_appear:
            LOGGER.info(
                "Online presence decision: skip%s",
                f" | {decision.reason}" if decision.reason else "",
            )
            return

        visible_seconds = random.randint(
            self._settings.online_presence_min_visible_seconds,
            self._settings.online_presence_max_visible_seconds,
        )
        await self._client(tl_functions.account.UpdateStatusRequest(offline=False))
        LOGGER.info(
            "Online presence pulse: on for %ss%s",
            visible_seconds,
            f" | {decision.reason}" if decision.reason else "",
        )
        self._last_online_pulse_at = now
        await asyncio.sleep(visible_seconds)
        await self._client(tl_functions.account.UpdateStatusRequest(offline=True))
        LOGGER.info("Online presence pulse: off")

    async def _maybe_send_proactive_message(self) -> None:
        if self._is_sleep_time_moscow():
            return
        if self._last_interactive_event_at is not None:
            if datetime.now().astimezone() - self._last_interactive_event_at < timedelta(minutes=5):
                return
        candidates: list[tuple[int, ApiChat, str]] = []
        for index, chat in enumerate(await self._iter_candidate_chats(), start=1):
            if not self._is_proactive_chat(chat.chat_label):
                continue
            last_message = self._store.get_last_message(chat.chat_id)
            if last_message is None:
                continue
            if self._looks_like_service_message(last_message.text):
                continue
            now_utc = datetime.utcnow()
            last_at = self._parse_store_timestamp(last_message.created_at)
            if now_utc - last_at < timedelta(minutes=self._settings.proactive_idle_minutes):
                continue
            last_outgoing = self._store.get_last_outgoing_message(chat.chat_id)
            if last_outgoing is not None:
                last_outgoing_at = self._parse_store_timestamp(last_outgoing.created_at)
                if now_utc - last_outgoing_at < timedelta(
                    minutes=self._settings.proactive_cooldown_minutes
                ):
                    continue
            if last_message.direction == "outgoing":
                continue

            history = self._store.get_recent_messages(chat.chat_id, min(6, self._settings.chat_history_limit))
            if not history:
                continue
            summary = " | ".join(item.content[:80] for item in history[-4:])
            candidates.append(
                (
                    index,
                    chat,
                    f"{index}. {chat.chat_label} | last_message_at={last_message.created_at} | summary={summary}",
                )
            )

        if not candidates:
            return

        choice = await self._responder.choose_proactive_chat(
            candidates_text="\n".join(item[2] for item in candidates)
        )
        if not choice.should_open or choice.selected_index is None:
            LOGGER.info(
                "Proactive chat choice: skip%s",
                f" | {choice.reason}" if choice.reason else "",
            )
            return

        selected_chat: ApiChat | None = None
        for index, chat, _ in candidates:
            if index == choice.selected_index:
                selected_chat = chat
                break
        if selected_chat is None:
            LOGGER.info("Proactive chat choice returned unknown index: %s", choice.selected_index)
            return

        history = self._store.get_recent_messages(
            selected_chat.chat_id,
            self._settings.chat_history_limit,
        )
        diary_entries = self._select_diary_context(selected_chat.chat_id)
        if self._settings.auto_send:
            assert self._client is not None
            async with self._client.action(selected_chat.entity, "typing"):
                decision = await self._responder.decide_proactive_message(
                    chat_label=selected_chat.chat_label,
                    history=history,
                    diary_entries=diary_entries,
                    daily_plan_context=self._build_daily_plan_context(),
                )
        else:
            decision = await self._responder.decide_proactive_message(
                chat_label=selected_chat.chat_label,
                history=history,
                diary_entries=diary_entries,
                daily_plan_context=self._build_daily_plan_context(),
            )
        if not decision.should_send or not decision.message_text.strip():
            LOGGER.info(
                "Proactive decision for %s: skip%s",
                selected_chat.chat_label,
                f" | {decision.reason}" if decision.reason else "",
            )
            return

        LOGGER.info(
            "Proactive decision for %s: send%s",
            selected_chat.chat_label,
            f" | {decision.reason}" if decision.reason else "",
        )
        await self._send_reply(
            selected_chat,
            [decision.message_text.strip()],
            "[proactive]",
            StickerDecision(
                should_send=False,
                candidate_id=None,
                mode="none",
                reason="",
                raw_response={},
            ),
            source_message_id=None,
            revision=None,
        )

    async def _refresh_daily_diary_legacy(self) -> None:
        day_key = self._now_moscow().date().isoformat()
        message_count = self._store.get_message_count_for_day(day_key)

        existing_entry = self._store.get_diary_entry(day_key)
        markdown_path = self._diary_markdown_path(day_key)
        source_messages = self._store.get_day_messages(day_key, self._settings.diary_source_limit)

        if message_count < self._settings.diary_min_messages_for_entry:
            if existing_entry is not None and markdown_path.exists():
                return
            summary = self._build_fallback_diary_summary(day_key, source_messages)
            self._store.upsert_diary_entry(
                day_key=day_key,
                summary=summary,
                source_message_count=message_count,
            )
            self._write_diary_markdown(day_key, summary, source_messages)
            LOGGER.info("Diary markdown ensured for %s using %s messages", day_key, message_count)
            return

        if existing_entry is not None:
            new_messages = message_count - existing_entry.source_message_count
            if new_messages < self._settings.diary_min_new_messages:
                if not markdown_path.exists():
                    self._write_diary_markdown(day_key, existing_entry.summary, source_messages)
                    LOGGER.info("Diary markdown restored for %s", day_key)
                return

        summary = ""
        if source_messages:
            summary = await self._responder.generate_diary_entry(
                day_key=day_key,
                source_messages=source_messages,
                existing_entry=existing_entry,
            )
        if not summary:
            summary = self._build_fallback_diary_summary(day_key, source_messages)

        self._store.upsert_diary_entry(
            day_key=day_key,
            summary=summary,
            source_message_count=message_count,
        )
        self._write_diary_markdown(day_key, summary, source_messages)
        LOGGER.info("Diary updated for %s using %s messages", day_key, message_count)

    def _diary_markdown_path(self, day_key: str) -> Path:
        return self._settings.diary_markdown_dir / f"{day_key}.md"

    @staticmethod
    def _build_fallback_diary_summary(day_key: str, source_messages) -> str:
        if not source_messages:
            return f"{day_key}: за день пока нет сохранённых событий. Персона просто держит дневник открытым."
        important = []
        for item in source_messages[-12:]:
            speaker = item.sender_name or item.chat_label
            important.append(f"{item.created_at} [{item.chat_label}] {speaker}: {item.text}")
        return (
            f"{day_key}: автоматическая заметка по событиям дня. "
            "Модель не дала отдельную сводку, поэтому сохранены последние факты.\n"
            + "\n".join(f"- {line}" for line in important)
        )

    def _write_diary_markdown(
        self,
        day_key: str,
        summary: str,
        source_messages,
    ) -> None:
        lines = [f"# Diary {day_key}", "", "## Summary", "", summary.strip(), "", "## Events", ""]
        for item in source_messages[-60:]:
            speaker = item.sender_name or item.chat_label
            direction = "Assistant" if item.direction == "outgoing" else speaker
            lines.append(f"- `{item.created_at}` [{item.chat_label}] {direction}: {item.text}")
        path = self._diary_markdown_path(day_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    async def _refresh_diary_if_needed(self) -> None:
        window = self._last_completed_diary_hour()
        day_key = window["day_key"]
        hour_key = window["hour_key"]
        period_label = window["period_label"]
        start_at = window["start_at"]
        end_at = window["end_at"]
        message_count = self._store.get_message_count_between(start_at, end_at)
        source_messages = self._filter_diary_source_messages(
            self._store.get_messages_between(
                start_at,
                end_at,
                self._settings.diary_source_limit,
            )
        )
        existing_entry = self._store.get_diary_entry(hour_key)
        markdown_path = self._diary_markdown_path(day_key)

        if existing_entry is not None and existing_entry.source_message_count >= message_count:
            if self._hourly_diary_section_exists(markdown_path, hour_key):
                return

        summary = ""
        if source_messages:
            summary = await self._responder.generate_diary_entry(
                day_key=period_label,
                source_messages=source_messages,
                existing_entry=existing_entry,
            )
        if not summary:
            summary = self._build_fallback_hourly_diary_summary(period_label, source_messages)

        self._store.upsert_diary_entry(
            day_key=hour_key,
            summary=summary,
            source_message_count=message_count,
        )
        self._write_hourly_diary_markdown(
            day_key=day_key,
            hour_key=hour_key,
            period_label=period_label,
            summary=summary,
            source_messages=source_messages,
        )
        LOGGER.info("Diary hour updated for %s using %s messages", period_label, message_count)

    def _last_completed_diary_hour(self) -> dict[str, str]:
        hour_start = self._now_moscow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        hour_end = hour_start + timedelta(hours=1)
        return {
            "day_key": hour_start.date().isoformat(),
            "hour_key": f"{hour_start.date().isoformat()}T{hour_start.hour:02d}",
            "period_label": f"{hour_start:%Y-%m-%d %H:00}-{(hour_end - timedelta(minutes=1)):%H:59} MSK",
            "start_at": hour_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_at": hour_end.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def _build_fallback_hourly_diary_summary(period_label: str, source_messages) -> str:
        if not source_messages:
            return f"- {period_label}: no notable Telegram events."
        chats = []
        topics = []
        for item in source_messages[-20:]:
            if item.chat_label not in chats:
                chats.append(item.chat_label)
            text = (item.text or "").strip()
            if text and len(topics) < 3:
                topics.append(text[:70])
        chat_text = ", ".join(chats[:4]) or "Telegram"
        topic_text = "; ".join(topics) or "short activity"
        return f"- {period_label}: activity in {chat_text}. Main: {topic_text}"
        if not source_messages:
            return f"{period_label}: за этот час заметных событий в Telegram не было."
        important = []
        for item in source_messages[-12:]:
            speaker = item.sender_name or item.chat_label
            important.append(f"{item.created_at} [{item.chat_label}] {speaker}: {item.text}")
        return (
            f"{period_label}: автоматическая почасовая заметка. "
            "Модель не дала отдельную сводку, поэтому сохранены факты часа.\n"
            + "\n".join(f"- {line}" for line in important)
        )

    @staticmethod
    def _hourly_diary_section_exists(path: Path, hour_key: str) -> bool:
        if not path.exists():
            return False
        return f"<!-- hour:{hour_key}:start -->" in path.read_text(encoding="utf-8")

    def _write_hourly_diary_markdown(
        self,
        *,
        day_key: str,
        hour_key: str,
        period_label: str,
        summary: str,
        source_messages,
    ) -> None:
        block_lines = [
            f"<!-- hour:{hour_key}:start -->",
            f"## {period_label}",
            "",
            "### Brief summary",
            "",
            summary.strip() or "- No notable Telegram events.",
            "",
            f"<!-- hour:{hour_key}:end -->",
        ]
        block = "\n".join(block_lines).strip()

        path = self._diary_markdown_path(day_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# Diary {day_key}\n\n"
        if not path.exists():
            path.write_text(header + block + "\n", encoding="utf-8")
            return

        content = path.read_text(encoding="utf-8")
        if not content.startswith(f"# Diary {day_key}"):
            content = header + content.strip() + "\n"
        pattern = re.compile(
            rf"<!-- hour:{re.escape(hour_key)}:start -->.*?<!-- hour:{re.escape(hour_key)}:end -->",
            flags=re.DOTALL,
        )
        if pattern.search(content):
            content = pattern.sub(block, content)
        else:
            content = content.rstrip() + "\n\n" + block + "\n"
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return
        block_lines = [
            f"<!-- hour:{hour_key}:start -->",
            f"## {period_label}",
            "",
            "### Сводка",
            "",
            summary.strip(),
            "",
            "### События",
            "",
        ]
        for item in source_messages[-60:]:
            speaker = item.sender_name or item.chat_label
            direction = "Assistant" if item.direction == "outgoing" else speaker
            block_lines.append(f"- `{item.created_at}` [{item.chat_label}] {direction}: {item.text}")
        if not source_messages:
            block_lines.append("- Событий не было.")
        block_lines.append(f"<!-- hour:{hour_key}:end -->")
        block = "\n".join(block_lines).strip()

        path = self._diary_markdown_path(day_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# Diary {day_key}\n\n"
        if not path.exists():
            path.write_text(header + block + "\n", encoding="utf-8")
            return

        content = path.read_text(encoding="utf-8")
        if not content.startswith(f"# Diary {day_key}"):
            content = header + content.strip() + "\n"
        pattern = re.compile(
            rf"<!-- hour:{re.escape(hour_key)}:start -->.*?<!-- hour:{re.escape(hour_key)}:end -->",
            flags=re.DOTALL,
        )
        if pattern.search(content):
            content = pattern.sub(block, content)
        else:
            content = content.rstrip() + "\n\n" + block + "\n"
        path.write_text(content.rstrip() + "\n", encoding="utf-8")

    @staticmethod
    def _build_fallback_diary_summary(day_key: str, source_messages) -> str:
        if not source_messages:
            return "день был тихий, я почти ни с кем по-настоящему не сблизилась и просто жила в своих мыслях."
        topics = [item.text.strip() for item in source_messages[-12:] if (item.text or "").strip()]
        joined = "; ".join(topics[:4]).strip()
        if not joined:
            return "день прошёл спокойно, без чего-то по-настоящему важного для сердца."
        return f"за день у меня в голове остались в основном вот эти чувства и разговоры: {joined}"

    def _write_diary_markdown(
        self,
        day_key: str,
        summary: str,
        source_messages,
    ) -> None:
        path = self._diary_markdown_path(day_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render_diary_day_markdown(day_key), encoding="utf-8")

    @staticmethod
    def _build_fallback_hourly_diary_summary(period_label: str, source_messages) -> str:
        if not source_messages:
            return "я почти ни с кем не говорила и просто была в своих мыслях."
        topics = []
        for item in source_messages[-20:]:
            text = (item.text or "").strip()
            if text and len(topics) < 3:
                topics.append(text[:70])
        topic_text = "; ".join(topics).strip()
        if not topic_text:
            return "час прошёл тихо, без чего-то по-настоящему важного."
        return f"я весь этот час крутила в голове {topic_text}"

    @staticmethod
    def _hourly_diary_section_exists(path: Path, hour_key: str) -> bool:
        return path.exists()

    def _write_hourly_diary_markdown(
        self,
        *,
        day_key: str,
        hour_key: str,
        period_label: str,
        summary: str,
        source_messages,
    ) -> None:
        path = self._diary_markdown_path(day_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render_diary_day_markdown(day_key), encoding="utf-8")

    def _render_diary_day_markdown(self, day_key: str) -> str:
        entries = self._store.list_diary_entries_for_day(day_key)
        thoughts = [self._render_diary_thought(entry) for entry in entries]
        thoughts = [item for item in thoughts if item]
        if not thoughts:
            thoughts = ["я почти ни с кем не говорила и просто была в своих мыслях."]
        return "\n\n".join(thoughts).strip() + "\n"

    @staticmethod
    def _render_diary_thought(entry: DiaryEntry) -> str:
        raw_text = (entry.summary or "").strip()
        if not raw_text:
            return ""
        text = re.sub(r"^\s*-\s*", "", raw_text)
        text = re.sub(
            r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}-\d{2}:\d{2}\s+MSK:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        lowered = text.lower()
        if (
            "no notable telegram events" in lowered
            or "утекла" in lowered
            or "обложк" in lowered
            or "альбом" in lowered
            or "мир трендов" in lowered
            or "http://" in lowered
            or "https://" in lowered
            or "t.me/" in lowered
            or "промокод" in lowered
            or "розыгрыш" in lowered
            or "канал" in lowered
        ):
            text = "я почти ни с кем не говорила и просто была в своих мыслях."
        elif not re.search(r"\bя\b|\bмне\b|\bу меня\b", lowered):
            cleaned = text.rstrip(". ")
            if cleaned:
                text = f"я думала о том, что {cleaned.lower()}."
        hour_match = re.match(r"\d{4}-\d{2}-\d{2}T(\d{2})$", entry.entry_date)
        if not hour_match:
            return text
        hour = hour_match.group(1)
        if len(text) > 1:
            text = text[0].lower() + text[1:]
        else:
            text = text.lower()
        return f"в {hour}:00-{hour}:59 по мск {text}"

    def _is_allowed_chat(self, chat_label: str) -> bool:
        if not self._settings.allowed_chats:
            return True
        lowered = chat_label.lower()
        return any(token in lowered for token in self._settings.allowed_chats)

    def _is_proactive_chat(self, chat_label: str) -> bool:
        if not self._settings.proactive_enabled:
            return False
        if not self._settings.proactive_allowed_chats:
            return True
        lowered = chat_label.lower()
        return any(token in lowered for token in self._settings.proactive_allowed_chats)

    @staticmethod
    def _looks_like_service_message(text: str) -> bool:
        value = (text or "").strip().lower()
        if not value:
            return False
        if "login code" in value or "код входа" in value or "код для входа" in value:
            return True
        if "telegram code" in value or "verification code" in value:
            return True
        return False

    async def _decide_sticker(
        self,
        *,
        chat: ApiChat,
        latest_incoming_text: str,
        planned_reply_text: str,
        history,
        diary_entries: list[DiaryEntry],
    ) -> StickerDecision:
        if not self._settings.stickers_enabled:
            return StickerDecision(
                should_send=False,
                candidate_id=None,
                mode="none",
                reason="stickers disabled",
                raw_response={},
            )
        normalized_text = latest_incoming_text.strip().lower().replace("ё", "е")
        if (
            "?" in latest_incoming_text
            or any(
                token in normalized_text
                for token in ("фото", "фотк", "картин", "изображ", "посмотри", "глянь", "что на")
            )
        ):
            return StickerDecision(
                should_send=False,
                candidate_id=None,
                mode="none",
                reason="photo/question context",
                raw_response={},
            )
        candidates = await self._get_sticker_candidates()
        if not candidates:
            return StickerDecision(
                should_send=False,
                candidate_id=None,
                mode="none",
                reason="no stickers found",
                raw_response={},
            )
        return await self._responder.choose_sticker(
            chat_label=chat.chat_label,
            latest_incoming_text=latest_incoming_text,
            planned_reply_text=planned_reply_text,
            history=history,
            diary_entries=diary_entries,
            candidates=[item.candidate for item in candidates],
        )

    async def _get_sticker_candidates(self) -> list[StickerRuntimeCandidate]:
        assert self._client is not None
        now = datetime.utcnow()
        if (
            self._sticker_cache
            and self._sticker_cache_refreshed_at is not None
            and now - self._sticker_cache_refreshed_at
            < timedelta(minutes=self._settings.sticker_cache_ttl_minutes)
        ):
            return self._sticker_cache

        seen_ids: set[int] = set()
        candidates: list[StickerRuntimeCandidate] = []
        responses = []
        with suppress(Exception):
            responses.append(("faved", await self._client(tl_functions.messages.GetFavedStickersRequest(hash=0))))
        with suppress(Exception):
            responses.append(
                (
                    "recent",
                    await self._client(tl_functions.messages.GetRecentStickersRequest(hash=0)),
                )
            )

        for source, response in responses:
            for document in getattr(response, "stickers", []) or []:
                candidate = self._build_sticker_candidate(document, source)
                if candidate is None:
                    continue
                document_id = int(getattr(document, "id", 0) or 0)
                if document_id <= 0 or document_id in seen_ids:
                    continue
                seen_ids.add(document_id)
                candidates.append(candidate)
                if len(candidates) >= self._settings.sticker_candidate_limit:
                    break
            if len(candidates) >= self._settings.sticker_candidate_limit:
                break

        self._sticker_cache = candidates
        self._sticker_cache_refreshed_at = now
        return candidates

    def _build_sticker_candidate(
        self,
        document: Any,
        source: str,
    ) -> StickerRuntimeCandidate | None:
        if document is None or isinstance(document, DocumentEmpty):
            return None
        emoji = self._extract_sticker_emoji(document)
        if emoji is None:
            return None
        filename = ""
        for attr in getattr(document, "attributes", []) or []:
            if isinstance(attr, DocumentAttributeFilename):
                filename = (attr.file_name or "").strip()
                break
        summary_parts = [emoji]
        if filename:
            summary_parts.append(filename)
        summary_parts.append("favorite" if source == "faved" else "recently used")
        document_id = int(getattr(document, "id", 0) or 0)
        candidate = StickerCandidate(
            candidate_id=str(document_id),
            emoji=emoji,
            summary=" | ".join(part for part in summary_parts if part),
            source=source,
        )
        return StickerRuntimeCandidate(candidate=candidate, document=document)

    @staticmethod
    def _extract_sticker_emoji(document: Any) -> str | None:
        for attr in getattr(document, "attributes", []) or []:
            if isinstance(attr, DocumentAttributeSticker):
                return (attr.alt or "").strip() or "sticker"
        return None

    async def _send_sticker(
        self,
        chat: ApiChat,
        sticker_decision: StickerDecision,
    ) -> bool:
        assert self._client is not None
        selected = next(
            (
                item
                for item in await self._get_sticker_candidates()
                if item.candidate.candidate_id == sticker_decision.candidate_id
            ),
            None,
        )
        if selected is None:
            LOGGER.info("Sticker candidate not found anymore: %s", sticker_decision.candidate_id)
            return False
        sent = await self._client.send_file(chat.entity, selected.document)
        self._store.add_message(
            chat_id=chat.chat_id,
            message_id=sent.id,
            direction="outgoing",
            text=f"[Sticker] {selected.candidate.emoji}",
            chat_label=chat.chat_label,
        )
        return True

    @staticmethod
    def _format_user_label(entity: User) -> str:
        first_name = (getattr(entity, "first_name", None) or "").strip()
        last_name = (getattr(entity, "last_name", None) or "").strip()
        username = (getattr(entity, "username", None) or "").strip()
        full_name = " ".join(part for part in [first_name, last_name] if part).strip()
        return full_name or username or str(getattr(entity, "id", "unknown"))

    @staticmethod
    def _parse_store_timestamp(value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
