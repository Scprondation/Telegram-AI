from __future__ import annotations

import asyncio
import base64
from difflib import SequenceMatcher
import hashlib
import logging
import random
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Page

from app.config import Settings, load_settings
from app.responder import ImageInput, LLMResponder
from app.storage import DiaryEntry, HistoryStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("telegram-web-assistant")

READY_STATE_JS = """
() => {
  const composerSelectors = [
    'div[contenteditable="true"][role="textbox"]',
    'div[contenteditable="true"][data-tab]',
    'div.input-message-input[contenteditable="true"]',
    'div[contenteditable="true"]'
  ];
  const titleSelectors = [
    'header h3',
    'header .title',
    'header [dir="auto"]',
    '.chat-info-wrapper .title',
    '.topbar h3'
  ];

  const findFirst = (selectors, root = document) => {
    for (const selector of selectors) {
      const element = root.querySelector(selector);
      if (element) {
        return element;
      }
    }
    return null;
  };

  const composer = findFirst(composerSelectors);
  if (!composer) {
    return null;
  }

  const titleElement = findFirst(titleSelectors);
  let chatLabel = (titleElement?.innerText || '').trim();
  if (!chatLabel) {
    chatLabel = (document.title || '').replace(/\\s*-\\s*Telegram.*$/i, '').trim();
  }
  if (!chatLabel) {
    return null;
  }

  const composerRect = composer.getBoundingClientRect();
  const titleRect = titleElement?.getBoundingClientRect();
  const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;

  // Focus on the lower dialogue region near the composer.
  // This avoids the header, the left chat list, and other UI chrome that confuses OCR.
  const leftEdge = Math.max(0, composerRect.left - 24);
  const rightEdge = Math.min(viewportWidth, composerRect.right + 96);
  const recentMessagesHeight = Math.min(
    Math.max(320, viewportHeight * 0.58),
    620
  );
  const titleBottom = titleRect?.bottom ?? 0;
  const topEdge = Math.max(
    titleBottom + 8,
    composerRect.top - recentMessagesHeight
  );
  const bottomEdge = Math.min(viewportHeight, composerRect.bottom + 24);

  const clip = {
    x: Math.max(0, leftEdge),
    y: Math.max(0, topEdge),
    width: Math.max(280, rightEdge - leftEdge),
    height: Math.max(240, bottomEdge - topEdge)
  };

  return {
    chat_label: chatLabel,
    input_text: (composer.innerText || composer.textContent || '').trim(),
    clip
  };
}
"""

COMPOSER_SELECTORS = [
    'div[contenteditable="true"][role="textbox"]',
    'div[contenteditable="true"][data-tab]',
    'div.input-message-input[contenteditable="true"]',
    'div[contenteditable="true"]',
]


@dataclass(slots=True)
class ChatFrame:
    chat_label: str
    chat_id: int
    input_text: str
    clip_x: float
    clip_y: float
    clip_width: float
    clip_height: float


class TelegramWebAssistant:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = HistoryStore(settings.history_db_path)
        self._responder = LLMResponder(settings)
        self._pending_tasks: dict[int, asyncio.Task[Any]] = {}
        self._diary_task: asyncio.Task[Any] | None = None
        self._proactive_task: asyncio.Task[Any] | None = None
        self._last_screenshot_hash_by_chat: dict[int, str] = {}
        self._last_handled_signature_by_chat: dict[int, str] = {}
        self._last_handled_text_by_chat: dict[int, str] = {}

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        async with async_playwright() as playwright:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(self._settings.browser_user_data_dir),
                headless=self._settings.browser_headless,
                channel=self._settings.browser_channel,
            )
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(self._settings.telegram_web_url, wait_until="domcontentloaded")
            await page.bring_to_front()

            LOGGER.info("Open Telegram Web, sign in if needed, and open a chat to monitor")
            frame = await self._wait_until_ready(page)
            LOGGER.info("Telegram Web is ready")
            LOGGER.info("Active chat detected: %s", frame.chat_label)
            LOGGER.info("Auto-send is %s", self._settings.auto_send)
            LOGGER.info("Diary memory is %s", self._settings.diary_enabled)
            LOGGER.info("Vision model: %s", self._settings.llm_model)
            LOGGER.info("Text model: %s", self._settings.text_llm_model)
            LOGGER.info("Proactive mode is %s", self._settings.proactive_enabled)

            if self._settings.diary_enabled:
                self._diary_task = asyncio.create_task(self._diary_loop())
            if self._settings.proactive_enabled:
                self._proactive_task = asyncio.create_task(self._proactive_loop(page))

            try:
                await self._poll_loop(page)
            finally:
                for task in list(self._pending_tasks.values()):
                    task.cancel()
                for task in list(self._pending_tasks.values()):
                    with suppress(asyncio.CancelledError):
                        await task
                if self._diary_task is not None:
                    self._diary_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._diary_task
                if self._proactive_task is not None:
                    self._proactive_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._proactive_task
                await context.close()

    async def _wait_until_ready(self, page: Page) -> ChatFrame:
        deadline = asyncio.get_running_loop().time() + self._settings.login_wait_timeout_seconds
        last_status_log_at = 0.0
        while asyncio.get_running_loop().time() < deadline:
            frame = await self._read_chat_frame(page)
            if frame is not None:
                return frame

            now = asyncio.get_running_loop().time()
            if now - last_status_log_at >= 10:
                last_status_log_at = now
                page_title = await page.title()
                LOGGER.info(
                    "Still waiting for Telegram Web readiness. Current page title: %s",
                    page_title,
                )
            await asyncio.sleep(2)
        raise RuntimeError(
            "Telegram Web did not become ready in time. Log in manually and open a chat."
        )

    async def _poll_loop(self, page: Page) -> None:
        while True:
            try:
                frame = await self._read_chat_frame(page)
                if frame is not None:
                    await self._handle_frame(page, frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Polling failed: %s", exc)

            await asyncio.sleep(self._settings.poll_interval_seconds)

    async def _handle_frame(self, page: Page, frame: ChatFrame) -> None:
        if not self._is_allowed_chat(frame.chat_label):
            return

        previous_task = self._pending_tasks.get(frame.chat_id)
        if previous_task is not None and not previous_task.done():
            return

        screenshot = await page.screenshot(
            type="jpeg",
            quality=55,
            clip={
                "x": frame.clip_x,
                "y": frame.clip_y,
                "width": frame.clip_width,
                "height": frame.clip_height,
            },
        )
        screenshot_hash = hashlib.sha1(screenshot).hexdigest()
        previous_hash = self._last_screenshot_hash_by_chat.get(frame.chat_id)
        if previous_hash == screenshot_hash:
            return
        self._last_screenshot_hash_by_chat[frame.chat_id] = screenshot_hash

        self._pending_tasks[frame.chat_id] = asyncio.create_task(
            self._analyze_and_maybe_reply(page, frame, screenshot, screenshot_hash)
        )

    async def _analyze_and_maybe_reply(
        self,
        page: Page,
        frame: ChatFrame,
        screenshot_bytes: bytes,
        screenshot_hash: str,
    ) -> None:
        try:
            diary_entries = self._select_diary_context(frame.chat_id)
            analysis = await self._responder.analyze_chat_screenshot(
                fallback_chat_label=frame.chat_label,
                image_input=ImageInput(
                    mime_type="image/jpeg",
                    base64_data=base64.b64encode(screenshot_bytes).decode("ascii"),
                ),
            )
            self._write_debug_capture(
                frame=frame,
                screenshot_bytes=screenshot_bytes,
                screenshot_hash=screenshot_hash,
                latest_incoming_text=analysis.latest_incoming_text,
                latest_incoming_signature=analysis.latest_incoming_signature,
                should_reply=analysis.should_reply,
            )

            signature = analysis.latest_incoming_signature.strip()
            if not signature:
                signature = screenshot_hash

            previous_signature = self._last_handled_signature_by_chat.get(frame.chat_id)
            latest_incoming_text = analysis.latest_incoming_text.strip()
            if previous_signature == signature:
                return

            previous_text = self._last_handled_text_by_chat.get(frame.chat_id, "")
            if self._looks_like_same_message(previous_text, latest_incoming_text):
                self._last_handled_signature_by_chat[frame.chat_id] = signature
                self._last_handled_text_by_chat[frame.chat_id] = latest_incoming_text
                LOGGER.info(
                    "Skipped duplicate OCR variant in %s: %s",
                    analysis.chat_label,
                    latest_incoming_text or "<empty>",
                )
                return

            if self._settings.trigger_prefix:
                if not latest_incoming_text.startswith(self._settings.trigger_prefix):
                    LOGGER.info(
                        "Latest message in %s ignored because prefix is missing",
                        analysis.chat_label,
                    )
                    self._last_handled_signature_by_chat[frame.chat_id] = signature
                    self._last_handled_text_by_chat[frame.chat_id] = latest_incoming_text
                    return
                stripped = latest_incoming_text[len(self._settings.trigger_prefix) :].strip()
                if stripped:
                    latest_incoming_text = stripped

            if not analysis.should_reply:
                LOGGER.info(
                    "No reply generated for %s | latest: %s",
                    analysis.chat_label,
                    latest_incoming_text or "<empty>",
                )
                self._last_handled_signature_by_chat[frame.chat_id] = signature
                self._last_handled_text_by_chat[frame.chat_id] = latest_incoming_text
                return

            self._store.add_message(
                chat_id=frame.chat_id,
                message_id=self._stable_int(f"{analysis.chat_label}:incoming:{signature}"),
                direction="incoming",
                text=latest_incoming_text or "[Unread incoming message]",
                chat_label=analysis.chat_label,
            )

            history = self._store.get_recent_messages(frame.chat_id, self._settings.chat_history_limit)
            reply = await self._responder.generate_text_reply(
                chat_label=analysis.chat_label,
                latest_incoming_text=latest_incoming_text,
                history=history,
                diary_entries=diary_entries,
            )
            reply_text = reply.text.strip() or analysis.fallback_reply.strip()
            if not reply_text:
                LOGGER.info(
                    "Reply generation returned empty text for %s",
                    analysis.chat_label,
                )
                self._last_handled_signature_by_chat[frame.chat_id] = signature
                return

            fresh_frame = await self._read_chat_frame(page)
            if fresh_frame is None or fresh_frame.chat_id != frame.chat_id:
                LOGGER.info("Reply cancelled because the active chat changed")
                return

            self._last_handled_signature_by_chat[frame.chat_id] = signature
            self._last_handled_text_by_chat[frame.chat_id] = latest_incoming_text
            await self._send_reply(page, fresh_frame, reply_text, latest_incoming_text)
        except asyncio.CancelledError:
            LOGGER.info("Screenshot analysis cancelled for %s", frame.chat_label)
            raise
        except Exception as exc:
            LOGGER.exception("Failed to analyze screenshot for %s: %s", frame.chat_label, exc)

    async def _send_reply(
        self,
        page: Page,
        frame: ChatFrame,
        reply_text: str,
        latest_incoming_text: str,
    ) -> None:
        if not self._settings.auto_send:
            LOGGER.info("Latest incoming in %s: %s", frame.chat_label, latest_incoming_text)
            LOGGER.info("Draft for %s: %s", frame.chat_label, reply_text)
            return

        composer = await self._find_composer(page)
        if composer is None:
            raise RuntimeError("Could not find the Telegram input box")

        if self._settings.skip_if_input_not_empty:
            current_input_text = (await composer.inner_text()).strip()
            if current_input_text:
                LOGGER.info("Skipped sending because the input already contains text")
                return

        await composer.click()
        delay = random.uniform(0.02, 0.08)
        for index, line in enumerate(reply_text.splitlines() or [reply_text]):
            if index > 0:
                await page.keyboard.down("Shift")
                await page.keyboard.press("Enter")
                await page.keyboard.up("Shift")
            if line:
                await page.keyboard.type(line, delay=delay)
        await asyncio.sleep(random.uniform(1.0, 2.5))
        await page.keyboard.press("Enter")

        self._store.add_message(
            chat_id=frame.chat_id,
            message_id=self._stable_int(f"{frame.chat_label}:outgoing:{datetime.now().isoformat()}"),
            direction="outgoing",
            text=reply_text,
            chat_label=frame.chat_label,
        )
        LOGGER.info("Reply sent to %s", frame.chat_label)

    async def _find_composer(self, page: Page):
        for selector in COMPOSER_SELECTORS:
            locator = page.locator(selector).last
            try:
                if await locator.count() > 0 and await locator.is_visible():
                    return locator
            except Exception:
                continue
        return None

    async def _read_chat_frame(self, page: Page) -> ChatFrame | None:
        payload = await page.evaluate(READY_STATE_JS)
        if payload is None:
            return None

        chat_label = str(payload.get("chat_label", "")).strip()
        if not chat_label:
            return None

        clip = payload.get("clip") or {}
        width = float(clip.get("width", 0))
        height = float(clip.get("height", 0))
        if width <= 0 or height <= 0:
            return None

        return ChatFrame(
            chat_label=chat_label,
            chat_id=self._stable_int(chat_label),
            input_text=str(payload.get("input_text", "")).strip(),
            clip_x=float(clip.get("x", 0)),
            clip_y=float(clip.get("y", 0)),
            clip_width=width,
            clip_height=height,
        )

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

    async def _diary_loop(self) -> None:
        while True:
            try:
                await self._refresh_diary_if_needed()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Diary refresh failed: %s", exc)
            await asyncio.sleep(self._settings.diary_check_interval_seconds)

    async def _proactive_loop(self, page: Page) -> None:
        while True:
            try:
                frame = await self._read_chat_frame(page)
                if frame is not None:
                    await self._maybe_send_proactive_message(page, frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Proactive loop failed: %s", exc)
            await asyncio.sleep(self._settings.proactive_check_interval_seconds)

    async def _maybe_send_proactive_message(self, page: Page, frame: ChatFrame) -> None:
        if not self._is_proactive_chat(frame.chat_label):
            return

        if self._pending_tasks.get(frame.chat_id) and not self._pending_tasks[frame.chat_id].done():
            return

        last_message = self._store.get_last_message(frame.chat_id)
        if last_message is None:
            return

        now_utc = datetime.utcnow()
        last_at = self._parse_store_timestamp(last_message.created_at)
        if now_utc - last_at < timedelta(minutes=self._settings.proactive_idle_minutes):
            return

        last_outgoing = self._store.get_last_outgoing_message(frame.chat_id)
        if last_outgoing is not None:
            last_outgoing_at = self._parse_store_timestamp(last_outgoing.created_at)
            if now_utc - last_outgoing_at < timedelta(
                minutes=self._settings.proactive_cooldown_minutes
            ):
                return

        if last_message.direction == "outgoing":
            return

        history = self._store.get_recent_messages(frame.chat_id, self._settings.chat_history_limit)
        if not history:
            return

        diary_entries = self._select_diary_context(frame.chat_id)
        decision = await self._responder.decide_proactive_message(
            chat_label=frame.chat_label,
            history=history,
            diary_entries=diary_entries,
        )
        if not decision.should_send or not decision.message_text.strip():
            LOGGER.info(
                "Proactive decision for %s: skip%s",
                frame.chat_label,
                f" | {decision.reason}" if decision.reason else "",
            )
            return

        fresh_frame = await self._read_chat_frame(page)
        if fresh_frame is None or fresh_frame.chat_id != frame.chat_id:
            LOGGER.info("Proactive message cancelled because the active chat changed")
            return

        LOGGER.info(
            "Proactive decision for %s: send%s",
            frame.chat_label,
            f" | {decision.reason}" if decision.reason else "",
        )
        await self._send_reply(
            page,
            fresh_frame,
            decision.message_text.strip(),
            "[proactive]",
        )

    async def _refresh_diary_if_needed(self) -> None:
        day_key = datetime.now().astimezone().date().isoformat()
        message_count = self._store.get_message_count_for_day(day_key)
        if message_count < self._settings.diary_min_messages_for_entry:
            return

        existing_entry = self._store.get_diary_entry(day_key)
        if existing_entry is not None:
            new_messages = message_count - existing_entry.source_message_count
            if new_messages < self._settings.diary_min_new_messages:
                return

        source_messages = self._store.get_day_messages(
            day_key,
            self._settings.diary_source_limit,
        )
        if not source_messages:
            return

        summary = await self._responder.generate_diary_entry(
            day_key=day_key,
            source_messages=source_messages,
            existing_entry=existing_entry,
        )
        if not summary:
            return

        self._store.upsert_diary_entry(
            day_key=day_key,
            summary=summary,
            source_message_count=message_count,
        )
        LOGGER.info("Diary updated for %s using %s messages", day_key, message_count)

    def _is_allowed_chat(self, chat_label: str) -> bool:
        if not self._settings.allowed_chats:
            return True
        lowered = chat_label.lower()
        return any(token in lowered for token in self._settings.allowed_chats)

    def _is_proactive_chat(self, chat_label: str) -> bool:
        if not self._settings.proactive_enabled:
            return False
        if not self._settings.proactive_allowed_chats:
            return False
        lowered = chat_label.lower()
        return any(token in lowered for token in self._settings.proactive_allowed_chats)

    def _write_debug_capture(
        self,
        *,
        frame: ChatFrame,
        screenshot_bytes: bytes,
        screenshot_hash: str,
        latest_incoming_text: str,
        latest_incoming_signature: str,
        should_reply: bool,
    ) -> None:
        if not self._settings.debug_screenshots_enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_chat = self._safe_filename(frame.chat_label)
        base_name = f"{timestamp}-{safe_chat}-{screenshot_hash[:10]}"
        image_path = self._settings.debug_dir / f"{base_name}.jpg"
        meta_path = self._settings.debug_dir / f"{base_name}.txt"

        image_path.write_bytes(screenshot_bytes)
        meta_text = (
            f"chat_label: {frame.chat_label}\n"
            f"chat_id: {frame.chat_id}\n"
            f"screenshot_hash: {screenshot_hash}\n"
            f"should_reply: {should_reply}\n"
            f"latest_incoming_signature: {latest_incoming_signature}\n"
            f"latest_incoming_text: {latest_incoming_text}\n"
        )
        meta_path.write_text(meta_text, encoding="utf-8")

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
        cleaned = cleaned.strip("_")
        return cleaned or "chat"

    @staticmethod
    def _looks_like_same_message(previous_text: str, current_text: str) -> bool:
        prev = TelegramWebAssistant._normalize_compare_text(previous_text)
        curr = TelegramWebAssistant._normalize_compare_text(current_text)
        if not prev or not curr:
            return False
        if prev == curr:
            return True
        if len(prev) >= 8 and len(curr) >= 8:
            return SequenceMatcher(a=prev, b=curr).ratio() >= 0.88
        return False

    @staticmethod
    def _normalize_compare_text(value: str) -> str:
        cleaned = "".join(ch.lower() for ch in value if ch.isalnum() or ch.isspace())
        return " ".join(cleaned.split())

    @staticmethod
    def _stable_int(value: str) -> int:
        digest = hashlib.sha1(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)

    @staticmethod
    def _parse_store_timestamp(value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def run() -> None:
    settings = load_settings()
    if settings.client_mode == "api":
        from app.api_runtime import TelegramApiAssistant

        assistant = TelegramApiAssistant(settings)
    elif settings.client_mode == "desktop":
        from app.desktop_runtime import TelegramDesktopAssistant

        assistant = TelegramDesktopAssistant(settings)
    else:
        assistant = TelegramWebAssistant(settings)

    try:
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user")
