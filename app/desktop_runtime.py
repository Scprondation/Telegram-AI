from __future__ import annotations

import asyncio
import base64
import ctypes
from difflib import SequenceMatcher
import hashlib
import io
import logging
import random
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta

import mss
import psutil
import pyperclip
from PIL import Image
from pywinauto import Desktop

from app.config import Settings
from app.responder import ImageInput, LLMResponder, SidebarChatChoice
from app.storage import DiaryEntry, HistoryStore


LOGGER = logging.getLogger("telegram-desktop-assistant")
USER32 = ctypes.windll.user32
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_MOUSEWHEEL = 0x020A
WM_PASTE = 0x0302
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
MK_LBUTTON = 0x0001
VK_RETURN = 0x0D


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


@dataclass(slots=True)
class DesktopFrame:
    chat_label: str
    chat_id: int
    window_handle: int
    clip_left: int
    clip_top: int
    clip_width: int
    clip_height: int
    sidebar_left: int
    sidebar_top: int
    sidebar_width: int
    sidebar_height: int
    input_x: int
    input_y: int


class TelegramDesktopAssistant:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = HistoryStore(settings.history_db_path)
        self._responder = LLMResponder(settings)
        self._pending_tasks: dict[int, asyncio.Task[None]] = {}
        self._diary_task: asyncio.Task[None] | None = None
        self._proactive_task: asyncio.Task[None] | None = None
        self._last_screenshot_hash_by_chat: dict[int, str] = {}
        self._last_handled_signature_by_chat: dict[int, str] = {}
        self._last_handled_text_by_chat: dict[int, str] = {}
        self._last_sidebar_hash: str = ""
        self._last_sidebar_choice_label: str = ""

    async def start(self) -> None:
        self._ensure_desktop_process_started()
        LOGGER.info("Open Telegram Desktop and sign in if needed")
        frame = await self._wait_until_ready()
        LOGGER.info("Telegram Desktop is ready")
        LOGGER.info(
            "Chat selection mode: %s",
            "auto-pick" if self._settings.desktop_auto_pick_chat else (self._settings.desktop_chat_label or "manual"),
        )
        LOGGER.info("Auto-send is %s", self._settings.auto_send)
        LOGGER.info("Diary memory is %s", self._settings.diary_enabled)
        LOGGER.info("Vision model: %s", self._settings.llm_model)
        LOGGER.info("Text model: %s", self._settings.text_llm_model)
        LOGGER.info("Proactive mode is %s", self._settings.proactive_enabled)

        if self._settings.diary_enabled:
            self._diary_task = asyncio.create_task(self._diary_loop())
        if self._settings.proactive_enabled:
            self._proactive_task = asyncio.create_task(self._proactive_loop())

        try:
            await self._poll_loop()
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

    async def _wait_until_ready(self) -> DesktopFrame:
        deadline = asyncio.get_running_loop().time() + self._settings.login_wait_timeout_seconds
        last_status_log_at = 0.0
        while asyncio.get_running_loop().time() < deadline:
            self._ensure_desktop_process_started()
            frame = self._read_chat_frame()
            if frame is not None:
                return frame

            now = asyncio.get_running_loop().time()
            if now - last_status_log_at >= 10:
                last_status_log_at = now
                LOGGER.info("Still waiting for Telegram Desktop window")
            await asyncio.sleep(2)
        raise RuntimeError(
            "Telegram Desktop window did not become ready in time. Log in manually and open a chat."
        )

    async def _poll_loop(self) -> None:
        while True:
            try:
                frame = self._read_chat_frame()
                if frame is not None:
                    await self._handle_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Desktop polling failed: %s", exc)
            await asyncio.sleep(self._settings.poll_interval_seconds)

    async def _handle_frame(self, frame: DesktopFrame) -> None:
        selected_label = await self._ensure_target_chat(frame)
        if not selected_label:
            return
        frame = self._clone_frame_with_chat(frame, selected_label)
        if not self._is_allowed_chat(frame.chat_label):
            return

        previous_task = self._pending_tasks.get(frame.chat_id)
        if previous_task is not None and not previous_task.done():
            return

        screenshot = self._capture_frame(frame)
        screenshot_hash = hashlib.sha1(screenshot).hexdigest()
        previous_hash = self._last_screenshot_hash_by_chat.get(frame.chat_id)
        if previous_hash == screenshot_hash:
            return
        self._last_screenshot_hash_by_chat[frame.chat_id] = screenshot_hash

        self._pending_tasks[frame.chat_id] = asyncio.create_task(
            self._analyze_and_maybe_reply(frame, screenshot, screenshot_hash)
        )

    async def _analyze_and_maybe_reply(
        self,
        frame: DesktopFrame,
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
                raw_model_text=analysis.raw_model_text,
            )

            signature = analysis.latest_incoming_signature.strip() or screenshot_hash
            latest_incoming_text = analysis.latest_incoming_text.strip()
            previous_signature = self._last_handled_signature_by_chat.get(frame.chat_id)
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
                self._last_handled_signature_by_chat[frame.chat_id] = signature
                self._last_handled_text_by_chat[frame.chat_id] = latest_incoming_text
                return

            self._last_handled_signature_by_chat[frame.chat_id] = signature
            self._last_handled_text_by_chat[frame.chat_id] = latest_incoming_text
            await self._send_reply(frame, reply_text, latest_incoming_text)
        except asyncio.CancelledError:
            LOGGER.info("Desktop screenshot analysis cancelled for %s", frame.chat_label)
            raise
        except Exception as exc:
            LOGGER.exception("Failed to analyze desktop screenshot for %s: %s", frame.chat_label, exc)

    async def _ensure_target_chat(self, frame: DesktopFrame) -> str | None:
        sidebar_bytes = self._capture_sidebar(frame)
        sidebar_hash = hashlib.sha1(sidebar_bytes).hexdigest()
        previous_sidebar_hash = self._last_sidebar_hash

        if (
            not self._settings.desktop_auto_pick_chat
            and self._settings.desktop_chat_label
            and self._last_sidebar_hash == sidebar_hash
            and self._last_sidebar_choice_label.lower() == self._settings.desktop_chat_label.lower()
        ):
            return self._settings.desktop_chat_label

        choice = await self._choose_sidebar_target(frame, sidebar_bytes)
        if (
            not choice.should_open
            and not self._settings.desktop_auto_pick_chat
            and self._settings.desktop_chat_label
        ):
            LOGGER.info("Scanning sidebar for target chat: %s", self._settings.desktop_chat_label)
            choice = await self._scan_sidebar_for_target(frame)
        self._last_sidebar_hash = sidebar_hash

        if not choice.should_open or choice.click_y is None:
            if self._settings.desktop_auto_pick_chat:
                LOGGER.info(
                    "Sidebar auto-pick skipped%s",
                    f" | {choice.reason}" if choice.reason else "",
                )
                return None
            LOGGER.info(
                "Target chat not found in sidebar: %s%s",
                self._settings.desktop_chat_label or "manual",
                f" | {choice.reason}" if choice.reason else "",
            )
            return None

        target_label = choice.chat_label.strip() or self._settings.desktop_chat_label or ""
        click_y = max(12, min(choice.click_y, frame.sidebar_height - 12))
        should_click = (
            self._last_sidebar_choice_label.lower() != target_label.lower()
            or previous_sidebar_hash != sidebar_hash
        )
        if should_click:
            self._background_click(
                frame.window_handle,
                frame.sidebar_left + max(24, frame.sidebar_width // 2),
                frame.sidebar_top + click_y,
            )
            await asyncio.sleep(0.45)
            LOGGER.info(
                "Selected chat in sidebar: %s%s",
                target_label,
                f" | {choice.reason}" if choice.reason else "",
            )
        self._last_sidebar_choice_label = target_label
        return target_label

    async def _choose_sidebar_target(
        self,
        frame: DesktopFrame,
        sidebar_bytes: bytes,
    ) -> SidebarChatChoice:
        return await self._responder.choose_chat_from_sidebar(
            image_input=ImageInput(
                mime_type="image/jpeg",
                base64_data=base64.b64encode(sidebar_bytes).decode("ascii"),
            ),
            preferred_chat_label=self._settings.desktop_chat_label,
            auto_pick=self._settings.desktop_auto_pick_chat,
        )

    async def _scan_sidebar_for_target(self, frame: DesktopFrame) -> SidebarChatChoice:
        wheel_x = frame.sidebar_left + max(24, frame.sidebar_width // 2)
        wheel_y = frame.sidebar_top + max(24, frame.sidebar_height // 2)
        search_pattern = [480, 480, 480, 480, -480, -480, -480, -480, -480, -480, -480, -480]
        last_choice = SidebarChatChoice(
            should_open=False,
            chat_label="",
            click_y=None,
            reason="No matching chat found after sidebar scan",
            raw_model_text="",
            raw_response={},
        )
        for delta in search_pattern:
            self._background_wheel(frame.window_handle, wheel_x, wheel_y, delta)
            await asyncio.sleep(0.25)
            sidebar_bytes = self._capture_sidebar(frame)
            choice = await self._choose_sidebar_target(frame, sidebar_bytes)
            if choice.should_open and choice.click_y is not None:
                return choice
            last_choice = choice
        return last_choice

    async def _send_reply(
        self,
        frame: DesktopFrame,
        reply_text: str,
        latest_incoming_text: str,
    ) -> None:
        if not self._settings.auto_send:
            LOGGER.info("Latest incoming in %s: %s", frame.chat_label, latest_incoming_text)
            LOGGER.info("Draft for %s: %s", frame.chat_label, reply_text)
            return

        self._background_click(frame.window_handle, frame.input_x, frame.input_y)
        await asyncio.sleep(0.2)
        pyperclip.copy(reply_text)
        USER32.PostMessageW(frame.window_handle, WM_PASTE, 0, 0)
        await asyncio.sleep(random.uniform(1.0, 2.5))
        USER32.PostMessageW(frame.window_handle, WM_KEYDOWN, VK_RETURN, 0)
        USER32.PostMessageW(frame.window_handle, WM_KEYUP, VK_RETURN, 0)

        self._store.add_message(
            chat_id=frame.chat_id,
            message_id=self._stable_int(f"{frame.chat_label}:outgoing:{datetime.now().isoformat()}"),
            direction="outgoing",
            text=reply_text,
            chat_label=frame.chat_label,
        )
        LOGGER.info("Reply sent to %s", frame.chat_label)

    def _read_chat_frame(self) -> DesktopFrame | None:
        window = self._find_target_window()
        if window is None:
            return None

        rect = window.rectangle()
        width = int(rect.width())
        height = int(rect.height())
        if width < 500 or height < 500:
            return None

        sidebar_left = int(rect.left + 8)
        sidebar_top = int(rect.top + 72)
        sidebar_right = int(rect.left + width * 0.31)
        sidebar_bottom = int(rect.bottom - 18)
        sidebar_width = max(180, sidebar_right - sidebar_left)
        sidebar_height = max(240, sidebar_bottom - sidebar_top)

        # Right dialogue pane plus recent message area above the composer.
        clip_left = int(rect.left + width * 0.31)
        clip_top = int(rect.top + max(72, height * 0.18))
        clip_right = int(rect.right - 18)
        clip_bottom = int(rect.bottom - 18)
        clip_width = max(320, clip_right - clip_left)
        clip_height = max(260, clip_bottom - clip_top)

        # Input area near the bottom of the right pane.
        input_x = int(rect.left + width * 0.64)
        input_y = int(rect.bottom - 56)

        chat_label = self._settings.desktop_chat_label or "desktop-chat"
        return DesktopFrame(
            chat_label=chat_label,
            chat_id=self._stable_int(chat_label),
            window_handle=window.handle,
            clip_left=clip_left,
            clip_top=clip_top,
            clip_width=clip_width,
            clip_height=clip_height,
            sidebar_left=sidebar_left,
            sidebar_top=sidebar_top,
            sidebar_width=sidebar_width,
            sidebar_height=sidebar_height,
            input_x=input_x,
            input_y=input_y,
        )

    def _capture_frame(self, frame: DesktopFrame) -> bytes:
        with mss.mss() as sct:
            shot = sct.grab(
                {
                    "left": frame.clip_left,
                    "top": frame.clip_top,
                    "width": frame.clip_width,
                    "height": frame.clip_height,
                }
            )
        image = Image.frombytes("RGB", shot.size, shot.rgb)
        width, height = image.size
        image = image.crop(
            (
                int(width * 0.08),
                int(height * 0.02),
                max(int(width * 0.62), 220),
                max(int(height * 0.96), 220),
            )
        )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        return buffer.getvalue()

    def _capture_sidebar(self, frame: DesktopFrame) -> bytes:
        with mss.mss() as sct:
            shot = sct.grab(
                {
                    "left": frame.sidebar_left,
                    "top": frame.sidebar_top,
                    "width": frame.sidebar_width,
                    "height": frame.sidebar_height,
                }
            )
        image = Image.frombytes("RGB", shot.size, shot.rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        return buffer.getvalue()

    def _background_click(self, hwnd: int, screen_x: int, screen_y: int) -> None:
        point = POINT(screen_x, screen_y)
        USER32.ScreenToClient(hwnd, ctypes.byref(point))
        lparam = (point.y << 16) | (point.x & 0xFFFF)
        USER32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)
        USER32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
        USER32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)

    def _background_wheel(self, hwnd: int, screen_x: int, screen_y: int, delta: int) -> None:
        point = POINT(screen_x, screen_y)
        USER32.ScreenToClient(hwnd, ctypes.byref(point))
        lparam = (point.y << 16) | (point.x & 0xFFFF)
        wparam = (ctypes.c_ushort(delta).value << 16)
        USER32.PostMessageW(hwnd, WM_MOUSEWHEEL, wparam, lparam)

    @staticmethod
    def _clone_frame_with_chat(frame: DesktopFrame, chat_label: str) -> DesktopFrame:
        return DesktopFrame(
            chat_label=chat_label,
            chat_id=TelegramDesktopAssistant._stable_int(chat_label),
            window_handle=frame.window_handle,
            clip_left=frame.clip_left,
            clip_top=frame.clip_top,
            clip_width=frame.clip_width,
            clip_height=frame.clip_height,
            sidebar_left=frame.sidebar_left,
            sidebar_top=frame.sidebar_top,
            sidebar_width=frame.sidebar_width,
            sidebar_height=frame.sidebar_height,
            input_x=frame.input_x,
            input_y=frame.input_y,
        )

    def _find_target_window(self):
        matches = []
        for window in Desktop(backend="win32").windows():
            try:
                pid = window.process_id()
                process_name = psutil.Process(pid).name().strip().lower()
                process_name = process_name.removesuffix(".exe")
                title = (window.window_text() or "").strip().lower()
                rect = window.rectangle()
            except Exception:
                continue
            if process_name not in self._settings.desktop_process_names:
                continue
            if title in {"", "ayugramdesktop", "qtrayiconmessagewindow"}:
                continue
            if "просмотр медиа" in title or "media viewer" in title:
                continue
            if rect.width() < 300 or rect.height() < 400:
                continue
            matches.append((rect.width() * rect.height(), window))
        if not matches:
            return None
        matches.sort(reverse=True, key=lambda item: item[0])
        return matches[0][1]

    def _ensure_desktop_process_started(self) -> None:
        if not self._settings.desktop_auto_launch:
            return
        if self._has_matching_process():
            return

        executable = self._settings.desktop_executable_path
        if executable is None or not executable.exists():
            return

        try:
            subprocess.Popen(
                [str(executable)],
                cwd=str(executable.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                close_fds=True,
            )
            LOGGER.info("Launched Telegram Desktop: %s", executable)
        except Exception as exc:
            LOGGER.exception("Failed to launch Telegram Desktop: %s", exc)

    def _has_matching_process(self) -> bool:
        target_names = self._settings.desktop_process_names
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").strip().lower().removesuffix(".exe")
            except Exception:
                continue
            if name in target_names:
                return True
        return False

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

    async def _proactive_loop(self) -> None:
        while True:
            try:
                frame = self._read_chat_frame()
                if frame is not None:
                    selected_label = await self._ensure_target_chat(frame)
                    if not selected_label:
                        await asyncio.sleep(self._settings.proactive_check_interval_seconds)
                        continue
                    frame = self._clone_frame_with_chat(frame, selected_label)
                    await self._maybe_send_proactive_message(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Proactive loop failed: %s", exc)
            await asyncio.sleep(self._settings.proactive_check_interval_seconds)

    async def _maybe_send_proactive_message(self, frame: DesktopFrame) -> None:
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
        LOGGER.info(
            "Proactive decision for %s: send%s",
            frame.chat_label,
            f" | {decision.reason}" if decision.reason else "",
        )
        await self._send_reply(frame, decision.message_text.strip(), "[proactive]")

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

        source_messages = self._store.get_day_messages(day_key, self._settings.diary_source_limit)
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
        frame: DesktopFrame,
        screenshot_bytes: bytes,
        screenshot_hash: str,
        latest_incoming_text: str,
        latest_incoming_signature: str,
        should_reply: bool,
        raw_model_text: str,
    ) -> None:
        if not self._settings.debug_screenshots_enabled:
            return
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_chat = self._safe_filename(frame.chat_label)
        base_name = f"{timestamp}-{safe_chat}-{screenshot_hash[:10]}"
        image_path = self._settings.debug_dir / f"{base_name}.jpg"
        meta_path = self._settings.debug_dir / f"{base_name}.txt"
        image_path.write_bytes(screenshot_bytes)
        meta_path.write_text(
            (
                f"chat_label: {frame.chat_label}\n"
                f"chat_id: {frame.chat_id}\n"
                f"screenshot_hash: {screenshot_hash}\n"
                f"should_reply: {should_reply}\n"
                f"latest_incoming_signature: {latest_incoming_signature}\n"
                f"latest_incoming_text: {latest_incoming_text}\n"
                f"raw_model_text: {raw_model_text}\n"
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
        cleaned = cleaned.strip("_")
        return cleaned or "chat"

    @staticmethod
    def _looks_like_same_message(previous_text: str, current_text: str) -> bool:
        prev = TelegramDesktopAssistant._normalize_compare_text(previous_text)
        curr = TelegramDesktopAssistant._normalize_compare_text(current_text)
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
