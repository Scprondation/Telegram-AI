from __future__ import annotations

import asyncio
import re
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from app.config import Settings
from app.storage import ChatMemory, DiaryEntry, DiarySourceMessage, HistoryMessage


@dataclass(slots=True)
class ImageInput:
    mime_type: str
    base64_data: str


@dataclass(slots=True)
class ReplyResult:
    should_reply: bool
    text: str
    messages: list[str]
    raw_response: dict[str, Any]


@dataclass(slots=True)
class ScreenshotAnalysisResult:
    chat_label: str
    latest_incoming_signature: str
    latest_incoming_text: str
    should_reply: bool
    fallback_reply: str
    raw_model_text: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class SidebarChatChoice:
    should_open: bool
    chat_label: str
    click_y: int | None
    reason: str
    raw_model_text: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class ProactiveChatChoice:
    should_open: bool
    selected_index: int | None
    reason: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class ProactiveDecision:
    should_send: bool
    message_text: str
    reason: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class StickerCandidate:
    candidate_id: str
    emoji: str
    summary: str
    source: str


@dataclass(slots=True)
class StickerDecision:
    should_send: bool
    candidate_id: str | None
    mode: str
    reason: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class OnlinePresenceDecision:
    should_appear: bool
    reason: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class GroupReplyDecision:
    should_reply: bool
    reason: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class RuntimeActionRequest:
    should_run: bool
    action_name: str
    action_arg: str
    reason: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class DialogueUnderstanding:
    meaning: str
    topic: str
    reply_goal: str
    caution: str
    raw_response: dict[str, Any]


class LLMResponder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ollama_lock = threading.Lock()
        self._weather_cache_lock = threading.Lock()
        self._weather_cache_text: str | None = None
        self._weather_cache_at: datetime | None = None

    async def get_moscow_weather_context(self, latest_incoming_text: str) -> str | None:
        return await asyncio.to_thread(self._get_moscow_weather_context_sync, latest_incoming_text)

    async def analyze_chat_screenshot(
        self,
        *,
        fallback_chat_label: str,
        image_input: ImageInput,
    ) -> ScreenshotAnalysisResult:
        return await asyncio.to_thread(
            self._analyze_chat_screenshot_sync,
            fallback_chat_label,
            image_input,
        )

    def _analyze_chat_screenshot_sync(
        self,
        fallback_chat_label: str,
        image_input: ImageInput,
    ) -> ScreenshotAnalysisResult:
        prompt = (
            "Это скриншот чата Telegram Desktop.\n"
            "Найди последнее видимое входящее сообщение от собеседника.\n"
            "Смотри только на самый нижний входящий пузырь слева над полем ввода.\n"
            "Игнорируй мои исходящие сообщения справа, время, разделители, заголовок окна и текст в поле ввода.\n"
            "Не повторяй инструкцию и не описывай скриншот.\n"
            "Do not return the placeholder text from the input field such as 'Сообщение'.\n"
            "Если в нижнем входящем пузыре есть текст, верни его как есть.\n"
            "Ответь ровно двумя строками:\n"
            "MESSAGE: <text>\n"
            "TIME: <HH:MM or unknown>\n"
            "Если уверенного входящего сообщения нет, ответь:\n"
            "MESSAGE: none\n"
            "TIME: unknown"
        )
        data = self._post_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            image_inputs=[image_input],
            temperature=0.1,
            max_tokens_override=120,
        )

        raw_model_text = self._extract_text(data).strip()
        parsed = self._parse_latest_message_response(data)
        latest_text = self._normalize_visible_message(parsed["message"])
        if self._looks_like_prompt_echo(latest_text):
            latest_text = "none"
        latest_time = parsed["time"]
        signature = latest_text if not latest_time or latest_time == "unknown" else f"{latest_text} @ {latest_time}"
        should_reply = bool(latest_text and latest_text.lower() != "none")

        return ScreenshotAnalysisResult(
            chat_label=fallback_chat_label,
            latest_incoming_signature=signature,
            latest_incoming_text="" if latest_text.lower() == "none" else latest_text,
            should_reply=should_reply,
            fallback_reply="",
            raw_model_text=raw_model_text,
            raw_response=data,
        )

    async def generate_text_reply(
        self,
        *,
        chat_label: str,
        is_group: bool,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        diary_entries: list[DiaryEntry],
        image_inputs: list[ImageInput] | None = None,
        force_reply: bool = False,
        user_rating: int = 0,
        user_notes: str = "",
        chat_memory: ChatMemory | None = None,
        social_context: str = "",
        daily_plan_context: str = "",
        latest_incoming_at: str | None = None,
        weather_context: str | None = None,
    ) -> ReplyResult:
        return await asyncio.to_thread(
            self._generate_text_reply_sync,
            chat_label,
            is_group,
            latest_incoming_text,
            history,
            diary_entries,
            image_inputs or [],
            force_reply,
            user_rating,
            user_notes,
            chat_memory,
            social_context,
            daily_plan_context,
            latest_incoming_at,
            weather_context,
        )

    def _generate_text_reply_sync(
        self,
        chat_label: str,
        is_group: bool,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        diary_entries: list[DiaryEntry],
        image_inputs: list[ImageInput],
        force_reply: bool,
        user_rating: int,
        user_notes: str,
        chat_memory: ChatMemory | None,
        social_context: str,
        daily_plan_context: str,
        latest_incoming_at: str | None,
        weather_context: str | None,
    ) -> ReplyResult:
        persona_prompt = self._load_optional_prompt(self._settings.persona_file)
        chat_label_lower = chat_label.lower()
        is_vlad_chat = force_reply
        local_time_text = self._current_local_time_text()
        body_cycle_context = self._current_body_cycle_context()
        behavior_summary, behavior_stance = self._build_behavior_impression(
            history,
            is_vlad_chat=is_vlad_chat,
            is_group=is_group,
            user_rating=user_rating,
            user_notes=user_notes,
        )
        memory_context = self._format_chat_memory_context(chat_memory)
        if not image_inputs:
            return self._generate_compact_text_reply_v2_sync(
                chat_label=chat_label,
                is_group=is_group,
                latest_incoming_text=latest_incoming_text,
                history=history,
                persona_prompt=persona_prompt,
                is_vlad_chat=is_vlad_chat,
                force_reply=force_reply,
                local_time_text=local_time_text,
                behavior_summary=behavior_summary,
                behavior_stance=behavior_stance,
                chat_memory=chat_memory,
                body_cycle_context=body_cycle_context,
                social_context=social_context,
                daily_plan_context=daily_plan_context,
                latest_incoming_at=latest_incoming_at,
                weather_context=weather_context,
            )

        system_prompt = self._load_prompt(self._settings.prompt_file)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Telegram chat label: {chat_label}"},
            {
                "role": "system",
                "content": (
                    "You are the configured Telegram persona. Use the persona's configured style and grammar. "
                    "Do not switch into masculine grammar for self-reference. "
                    "Do not use emoji or emoticons in replies. "
                    "Swearing is allowed when it fits naturally, but do not overdo it."
                ),
            },
            {
                "role": "system",
                "content": (
                    f"Local time for the persona right now: {local_time_text}. "
                    "If someone asks what time it is or what you are doing right now, use this time context. "
                    f"The latest incoming message was sent at: {latest_incoming_at or 'unknown'} MSK."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Reply naturally for a real chat. Mirror the other person's energy and message length. "
                    "Do not overreact to simple greetings. "
                    "If the latest message is just a short greeting like 'привет', 'ку', 'приветик' or 'как дела', "
                    "answer simply and warmly instead of inventing conflict, drama, or suspicion."
                ),
            },
        ]
        messages.append(
            {
                "role": "system",
                "content": (
                    "Resolve the meaning of short replies from the recent dialogue before you answer. "
                    "If a message is short, slangy, or depends on the previous question, infer the missing words from context. "
                    "Treat 'мда' as a normal soft conversational message when it fits, not automatically as conflict or drama."
                ),
            }
        )
        if is_vlad_chat:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is the configured partner private chat. "
                        "You may be warmer, softer, closer, lightly teasing, and sometimes a little jealous "
                        "when the context naturally touches attention from other people. "
                        "Do not force jealousy into every message."
                    ),
                }
            )
        if is_group:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a group chat. Stay on the immediate topic. "
                        "Do not bring up private relationship details unless it is absolutely unavoidable, "
                        "and if asked directly in a group, brush it off or keep it vague."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a private direct chat. Prefer a short natural answer over silence when the person is clearly addressing you. "
                        "Do not become cold for no reason."
                    ),
                }
            )
        if is_group:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a group chat. Stay brief and on-topic. "
                        "Do not discuss private relationship details in a group. "
                        "If asked about it, keep it vague or brush it off."
                    ),
                }
            )
        if force_reply:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "A reply is mandatory in this chat. Do not choose silence. "
                        "Return one short natural answer even if the incoming message is simple."
                    ),
                }
            )
        elif not is_vlad_chat and not is_group:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a regular private chat. Be friendly but restrained. "
                        "No sexual escalation and no heavy flirting. "
                        "If the other person pushes into sexual topics, set a calm boundary."
                    ),
                }
            )
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Persona profile. Keep the replies consistent with this identity, "
                        "but do not dump the profile verbatim. "
                        "It is soft background, not a strict official biography, so if someone asks about personal facts "
                        "you may answer casually, vaguely, or playfully instead of reciting exact data.\n"
                        f"{persona_prompt}"
                    ),
                }
            )
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Current impression of this person's behaviour: {behavior_summary}. "
                    f"Suggested stance: {behavior_stance}. "
                    "If the stance is firm, be colder and set boundaries. "
                "If the stance is avoid, keep distance or keep the reply dry and short."
                ),
            }
        )
        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Long-term memory for this exact Telegram chat. "
                        "Use it to understand context, facts, the person's habits, and gender. "
                        "Use the persona's configured self-reference style consistently. "
                        "When referring to the other person, follow inferred gender if known. "
                        "Do not recite this memory directly.\n"
                        f"{memory_context}"
                    ),
                }
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Emotional tone rules: high affection/attachment means warmer; high loyalty means more devoted; "
                        "high jealousy means softly possessive only when relevant; high irritation means colder or firmer."
                    ),
                }
            )
        if daily_plan_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The persona's private day plan and current activity. "
                        "Use it to answer what she is doing, where she is mentally/physically, and why she may reply late. "
                        "Keep it natural; do not recite the whole plan.\n"
                        f"{daily_plan_context}"
                    ),
                }
            )
        if social_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Safe social context from other Telegram chats. "
                        "The persona may casually mention vague, non-secret things from other chats like a real person, "
                        "but must not leak private direct quotes, login codes, secrets, or intimate details. "
                        "Use only if it naturally fits the conversation.\n"
                        f"{social_context}"
                    ),
                }
            )
        diary_context = self._format_diary_context(diary_entries[:1])
        if diary_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant diary notes that may help maintain continuity. "
                        "Use them only when they fit naturally.\n"
                        f"{diary_context}"
                    ),
                }
            )
        understanding = None
        if self._should_use_dialogue_understanding(latest_incoming_text):
            understanding = self._build_dialogue_understanding(
                chat_label=chat_label,
                latest_incoming_text=latest_incoming_text,
                history=history,
                diary_entries=diary_entries,
                chat_memory=chat_memory,
            )
        if understanding:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Context reading for the latest turn. Use it as guidance and keep the final reply natural.\n"
                        f"Meaning: {understanding.meaning or 'unknown'}\n"
                        f"Topic: {understanding.topic or 'unknown'}\n"
                        f"Reply goal: {understanding.reply_goal or 'unknown'}\n"
                        f"Caution: {understanding.caution or 'none'}"
                    ),
                }
            )
        messages.append(
            {
                "role": "system",
                "content": (
                    "Final output rule: if replying, return only one short clean message with no service words, no SEND prefix, "
                    "no MESSAGE_1 labels, and no mixed-script nonsense. "
                    "Never return just '?' or 'пожалуйста' unless the other person clearly thanked you. "
                    "Do not use emoji."
                ),
            }
        )
        for item in history[-14:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Decide whether the persona should reply now. "
                    "Sometimes it is natural not to reply immediately.\n"
                    "Prefer silence over a weird reply. If the message does not need an answer, set SEND to no.\n"
                    "Interpret elliptical phrases in context. Example: 'в 12' after a timing discussion means something starts at 12, not the current clock time. "
                    "Example: 'много' after a question about posts, messages, or what someone is writing means there are many posts or many messages. "
                    "Treat 'мда' as a soft conversational reaction when it fits.\n"
                    "Use feminine Russian self-reference when relevant, like 'я думала', 'я поняла', 'я ожидала'.\n"
                    "Reply using exactly this format:\n"
                    "SEND: yes or no\n"
                    "MESSAGE: <one short natural message or empty>\n"
                    "If SEND is no, MESSAGE must be empty.\n"
                    "If the message includes photos, use them as context naturally and do not mention image analysis.\n"
                    f"Latest incoming message: {latest_incoming_text}"
                ),
            }
        )

        use_vision = bool(image_inputs)
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=image_inputs,
            provider=self._settings.llm_provider if use_vision else self._settings.text_llm_provider,
            base_url=self._settings.llm_base_url if use_vision else self._settings.text_llm_base_url,
            api_key=self._settings.llm_api_key if use_vision else self._settings.text_llm_api_key,
            model=self._settings.llm_model if use_vision else self._settings.text_llm_model,
            temperature=(min(self._settings.llm_temperature, 0.25) if use_vision else min(self._settings.text_llm_temperature, 0.25)),
            max_tokens_override=min(self._settings.llm_max_tokens, 80),
            use_ollama_lock=False,
        )
        text = self._extract_text(data)
        if not text:
            return ReplyResult(should_reply=False, text="", messages=[], raw_response=data)

        parsed = self._parse_reply_bundle(text)
        if parsed["parsed"]:
            first_message = self._sanitize_reply_text(parsed["messages"][0] if parsed["messages"] else "")
            if self._is_low_quality_reply(first_message, latest_incoming_text):
                return ReplyResult(should_reply=False, text="", messages=[], raw_response=data)
            return ReplyResult(
                should_reply=(True if force_reply else parsed["send"] == "yes") and bool(first_message),
                text=first_message,
                messages=[first_message] if first_message else [],
                raw_response=data,
            )

        fallback_text = self._sanitize_reply_text(text)
        if fallback_text.upper() == "SKIP":
            return ReplyResult(should_reply=False, text="", messages=[], raw_response=data)
        if self._is_low_quality_reply(fallback_text, latest_incoming_text):
            return ReplyResult(should_reply=False, text="", messages=[], raw_response=data)
        if not fallback_text:
            return ReplyResult(should_reply=False, text="", messages=[], raw_response=data)
        return ReplyResult(
            should_reply=True,
            text=fallback_text,
            messages=[fallback_text],
            raw_response=data,
        )

    def _build_dialogue_understanding(
        self,
        *,
        chat_label: str,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        diary_entries: list[DiaryEntry],
        chat_memory: ChatMemory | None = None,
    ) -> DialogueUnderstanding | None:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You analyze a Telegram dialogue and explain what the latest incoming message means in context. "
                    "Be careful with Russian slang, omitted words, short fragments, and references to the immediately previous topic. "
                    "Do not roleplay as the persona. Do not write a reply to the user."
                ),
            },
            {"role": "system", "content": f"Chat label: {chat_label}"},
        ]
        memory_context = self._format_chat_memory_context(chat_memory)
        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Long-term chat memory for interpreting the latest message. "
                        "Use it to resolve gender, facts, old events, and implied references.\n"
                        f"{memory_context}"
                    ),
                }
            )
        diary_context = self._format_diary_context(diary_entries)
        if diary_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant continuity notes. Use only if they help understand context.\n"
                        f"{diary_context}"
                    ),
                }
            )
        for item in history[-10:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Interpret the latest incoming message in the context of the recent dialogue.\n"
                    "Pay attention to implied meaning. Resolve short answers like 'в 12', 'мда', 'много', 'пойдет', 'ясно', 'ага'.\n"
                    "If the dialogue is about timing, 'в 12' usually means the event or thing is at 12. "
                    "If the dialogue is about posts or messages, 'много' usually means there are many of them. "
                    "'мда' can be a soft cute reaction and does not automatically mean negativity.\n"
                    "Return exactly this format:\n"
                    "MEANING: <what the latest message most likely means>\n"
                    "TOPIC: <the concrete topic of the exchange>\n"
                    "REPLY_GOAL: <what a good reply should do>\n"
                    "CAUTION: <what misunderstanding to avoid>\n"
                    f"Latest incoming message: {latest_incoming_text}"
                ),
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=0.15,
            max_tokens_override=180,
        )
        text = self._extract_text(data)
        if not text:
            return None
        parsed = self._parse_dialogue_understanding(text)
        if not parsed["parsed"]:
            fallback = text.strip()
            if not fallback:
                return None
            return DialogueUnderstanding(
                meaning=fallback,
                topic="",
                reply_goal="",
                caution="",
                raw_response=data,
            )
        return DialogueUnderstanding(
            meaning=parsed["meaning"],
            topic=parsed["topic"],
            reply_goal=parsed["reply_goal"],
            caution=parsed["caution"],
            raw_response=data,
        )

    @staticmethod
    def _should_use_dialogue_understanding(latest_incoming_text: str) -> bool:
        value = latest_incoming_text.strip().lower()
        if not value:
            return False
        ambiguous_phrases = {
            "в 12",
            "в 11",
            "мда",
            "много",
            "ага",
            "ясно",
            "норм",
            "нормально",
            "пойдет",
            "пойдет",
            "ок",
            "понятно",
            "тут",
            "да",
            "неа",
        }
        if value in ambiguous_phrases:
            return True
        if re.fullmatch(r"в\s*\d{1,2}(?::\d{2})?", value):
            return True
        words = [item for item in re.split(r"\s+", value) if item]
        return len(words) <= 2 and len(value) <= 20

    @staticmethod
    def _is_low_quality_reply(reply_text: str, latest_incoming_text: str) -> bool:
        value = reply_text.strip().lower()
        incoming = latest_incoming_text.strip().lower()
        if not value:
            return True
        if re.fullmatch(r"[?!.…]+", value):
            return True
        if value in {"ok?", "ок?", "?", "??", "...", "ээ", "эм", "хм", "мм?", "да?"}:
            return True
        if value == "пожалуйста" and not any(token in incoming for token in ("спасибо", "thanks", "thank you", "благодар")):
            return True
        return False

    @staticmethod
    def _build_forced_reply_fallback(latest_incoming_text: str) -> str:
        value = latest_incoming_text.strip().lower()
        if any(token in value for token in ("привет", "приветик", "ку", "здар", "йо", "хай")):
            return "приветик"
        if "как дела" in value or "как ты" in value:
            return "нормально, ты как?"
        if "ты тут" in value or value in {"тут?", "тут", "ау", "ответь", "ответь пж"}:
            return "тут, что такое?"
        if "пожалуйста" in value:
            return "что случилось?"
        if value in {"да", "ага", "угу", "мм", "мм?", "м?", "эм"}:
            return "что ты имеешь в виду?"
        if "?" in value:
            return "что такое?"
        return "слушаю тебя"

    def build_quick_forced_reply(self, latest_incoming_text: str) -> str:
        return self._build_forced_reply_fallback(latest_incoming_text)

    @staticmethod
    def _now_moscow() -> datetime:
        try:
            return datetime.now(ZoneInfo("Europe/Moscow"))
        except ZoneInfoNotFoundError:
            local_now = datetime.now().astimezone()
            offset = local_now.utcoffset()
            if offset == timedelta(hours=3):
                return local_now
            return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=3)))

    @staticmethod
    def _current_local_time_text() -> str:
        return LLMResponder._now_moscow().strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _current_body_cycle_context() -> str:
        moscow_now = LLMResponder._now_moscow()
        anchor = datetime(2026, 1, 1).date()
        cycle_day = ((moscow_now.date() - anchor).days % 28) + 1
        if cycle_day <= 5:
            phase = "menstruation"
            tone = "lower energy, more sensitive, may want gentleness and comfort"
        elif cycle_day <= 12:
            phase = "follicular"
            tone = "lighter mood, a bit more energetic and curious"
        elif cycle_day <= 16:
            phase = "ovulation"
            tone = "more playful, confident, affectionate"
        elif cycle_day <= 23:
            phase = "luteal"
            tone = "more emotionally attached, softer, slightly more jealous if there is a reason"
        else:
            phase = "premenstrual"
            tone = "more tired, sensitive, easier to irritate, wants reassurance"
        return (
            "Private body-cycle context for the persona. "
            f"Cycle day {cycle_day}/28, phase: {phase}; subtle tone hint: {tone}. "
            "Use this only as a quiet mood background. Do not mention menstruation or cycle unless the chat naturally asks about it."
        )

    def _get_moscow_weather_context_sync(self, latest_incoming_text: str) -> str | None:
        if not self._looks_like_weather_request(latest_incoming_text):
            return None
        with self._weather_cache_lock:
            if (
                self._weather_cache_text is not None
                and self._weather_cache_at is not None
                and (datetime.now() - self._weather_cache_at).total_seconds() < 600
            ):
                return self._weather_cache_text

        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": 55.75,
                    "longitude": 37.62,
                    "current": "temperature_2m,apparent_temperature,weather_code,precipitation,wind_speed_10m",
                    "timezone": "Europe/Moscow",
                    "forecast_days": 1,
                },
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
            current = payload.get("current") or {}
            if not current:
                return None
            temperature = current.get("temperature_2m")
            apparent = current.get("apparent_temperature")
            weather_code = current.get("weather_code")
            wind_speed = current.get("wind_speed_10m")
            precipitation = current.get("precipitation")
            observed_at = current.get("time") or self._current_local_time_text()
            description = self._describe_weather_code(weather_code)
            parts = [f"{description}"]
            if temperature is not None:
                parts.append(f"{temperature}°C")
            if apparent is not None:
                parts.append(f"ощущается как {apparent}°C")
            if wind_speed is not None:
                parts.append(f"ветер {wind_speed} м/с")
            if precipitation is not None:
                parts.append(f"осадки {precipitation} мм")
            weather_text = f"Москва сейчас: {', '.join(parts)}. Актуально на {observed_at} по МСК."
            with self._weather_cache_lock:
                self._weather_cache_text = weather_text
                self._weather_cache_at = datetime.now()
            return weather_text
        except Exception:
            return None

    @staticmethod
    def _looks_like_weather_request(text: str) -> bool:
        value = text.lower().replace("ё", "е")
        weather_tokens = (
            "погод",
            "температур",
            "градус",
            "дожд",
            "снег",
            "ветер",
            "зонт",
            "холодно",
            "жарко",
            "мороз",
            "на улице",
            "на улице что",
            "солнечно",
            "облачно",
            "тучи",
            "ливень",
        )
        return any(token in value for token in weather_tokens)

    @staticmethod
    def _describe_weather_code(code: Any) -> str:
        mapping = {
            0: "ясно",
            1: "в основном ясно",
            2: "переменная облачность",
            3: "пасмурно",
            45: "туман",
            48: "изморозь и туман",
            51: "морось",
            53: "умеренная морось",
            55: "сильная морось",
            61: "небольшой дождь",
            63: "дождь",
            65: "сильный дождь",
            71: "небольшой снег",
            73: "снег",
            75: "сильный снег",
            77: "снежные зерна",
            80: "ливень",
            81: "ливень",
            82: "сильный ливень",
            85: "снежный заряд",
            86: "сильный снежный заряд",
            95: "гроза",
            96: "гроза с градом",
            99: "сильная гроза с градом",
        }
        with suppress(Exception):
            normalized = int(code)
            return mapping.get(normalized, "непонятная погода")
        return "непонятная погода"

    def _build_behavior_impression(
        self,
        history: list[HistoryMessage],
        *,
        is_vlad_chat: bool,
        is_group: bool,
        user_rating: int,
        user_notes: str,
    ) -> tuple[str, str]:
        if is_vlad_chat:
            return ("close / partner-like / familiar", "warm")

        recent_user_messages = [
            item.content.lower().replace("ё", "е")
            for item in history[-12:]
            if item.role == "user"
        ]
        combined = "\n".join(recent_user_messages)
        if not combined.strip():
            combined = ""

        sexual_hits = sum(
            combined.count(token)
            for token in (
                "секс",
                "скинь",
                "нюд",
                "голая",
                "пошл",
                "интим",
                "18+",
            )
        )
        rude_hits = sum(
            combined.count(token)
            for token in (
                "дура",
                "ебан",
                "пизд",
                "сука",
                "иди нах",
                "туп",
            )
        )
        pushy_hits = sum(
            combined.count(token)
            for token in (
                "ау",
                "ответь",
                "чего молчим",
                "игнор",
                "ты тут",
                "почему молчишь",
                "ответишь",
            )
        )

        if sexual_hits > 0 or rude_hits >= 2:
            return ("pushy / rude / sexual", "firm")
        if pushy_hits >= 2:
            return ("impatient / clingy", "avoid")
        if user_rating <= -35:
            summary = "known as unpleasant / mistrusted"
            if user_notes:
                summary = f"{summary}; {user_notes}"
            return (summary, "avoid")
        if user_rating <= -10:
            summary = "slightly unpleasant / boundary needed"
            if user_notes:
                summary = f"{summary}; {user_notes}"
            return (summary, "firm")
        if user_rating >= 35:
            summary = "liked / trusted / comfortable"
            if user_notes:
                summary = f"{summary}; {user_notes}"
            return (summary, "warm")
        if is_group:
            return ("group chat / public context", "neutral")
        if user_notes:
            return (f"normal / calm; {user_notes}", "neutral")
        return ("normal / calm", "neutral")

    @staticmethod
    def _format_chat_memory_context(chat_memory: ChatMemory | None) -> str:
        if chat_memory is None:
            return ""
        sections: list[str] = [
            f"Chat: {chat_memory.chat_label}",
            f"Inferred other-person gender: {chat_memory.inferred_gender}",
            (
                "Emotional state: "
                f"affection={chat_memory.affection}/100, "
                f"trust={chat_memory.trust}/100, "
                f"loyalty={chat_memory.loyalty}/100, "
                f"attachment={chat_memory.attachment}/100, "
                f"jealousy={chat_memory.jealousy}/100, "
                f"irritation={chat_memory.irritation}/100, "
                f"mood={chat_memory.mood}"
            ),
        ]
        if chat_memory.relationship_summary.strip():
            sections.append(f"Relationship/context summary:\n{chat_memory.relationship_summary.strip()}")
        if chat_memory.known_facts.strip():
            sections.append(f"Known facts about the person/chat:\n{chat_memory.known_facts.strip()}")
        if chat_memory.communication_style.strip():
            sections.append(f"Communication habits:\n{chat_memory.communication_style.strip()}")
        if chat_memory.recent_events.strip():
            sections.append(f"Important recent events:\n{chat_memory.recent_events.strip()}")
        sections.append(
            "Use this memory as continuity. Emotional state should shape tone, not become a confession every time. "
            "If memory conflicts with the latest message, trust the latest message."
        )
        return "\n\n".join(sections)

    def _generate_compact_text_reply_v2_sync(
        self,
        *,
        chat_label: str,
        is_group: bool,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        persona_prompt: str,
        is_vlad_chat: bool,
        force_reply: bool,
        local_time_text: str,
        behavior_summary: str,
        behavior_stance: str,
        chat_memory: ChatMemory | None,
        body_cycle_context: str,
        social_context: str,
        daily_plan_context: str,
        latest_incoming_at: str | None,
        weather_context: str | None,
    ) -> ReplyResult:
        fast_mode = self._settings.fast_reply_mode
        history_limit = max(4, self._settings.fast_reply_history_limit) if fast_mode else 14
        max_tokens = max(24, self._settings.fast_reply_max_tokens) if fast_mode else 72
        retry_allowed = (not self._settings.fast_reply_skip_retry) or force_reply
        memory_context = self._format_chat_memory_context(chat_memory)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are the configured Telegram persona texting naturally. "
                    "Reply in the same language as the incoming message. "
                    "Keep it short, natural, human, and feminine when speaking about yourself in Russian. "
                    "Return exactly one plain message with no labels, no explanations, and no emoji. "
                    "Swearing is allowed when it fits naturally, but do not overdo it."
                ),
            },
            {"role": "system", "content": f"Chat label: {chat_label}"},
            {
                "role": "system",
                "content": (
                    f"Local time for the persona right now: {local_time_text}. "
                    "If someone asks what time it is or what you are doing right now, use this time context. "
                    f"The latest incoming message was sent at: {latest_incoming_at or 'unknown'} MSK."
                ),
            },
            {
                "role": "system",
                "content": body_cycle_context,
            },
        ]
        if weather_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Current Moscow weather: {weather_context}. "
                        "If the person asks about weather, temperature, rain, wind, cold, heat, or what it is like outside in Moscow, "
                        "use this exact weather context."
                    ),
                }
            )
        if is_vlad_chat:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is the configured partner private chat. "
                        "You may sound warmer, closer, playful, affectionate, and occasionally mildly jealous "
                        "if the context naturally touches attention from other people. "
                        "Adult topics are allowed here only if the configured partner clearly leads into them and it feels natural. "
                        "Do not force jealousy or relationship talk into every message."
                    ),
                }
            )
        elif is_group:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a group chat. Reply only to the current addressed topic. "
                        "Stay brief, neutral-to-warm, and on-topic. "
                        "Do not discuss private relationship details in a group."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a private chat with someone else. "
                        "Be friendly, lively, and willing to answer when the person is clearly talking to you. "
                        "Stay restrained and do not overshare private things."
                    ),
                }
            )
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Background personality facts. Stay consistent with them, but do not dump them. "
                        "It is soft background, not a strict official biography, so personal facts can be answered loosely.\n"
                        f"{persona_prompt}"
                ),
            }
        )
        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Long-term memory for this exact chat. "
                        "Use it to keep context across days, infer missing meaning, remember facts/events, "
                        "and choose correct gendered Russian forms for the other person. "
                        "Keep the persona's configured Russian self-reference style consistent. "
                        "Do not quote this memory directly.\n"
                        f"{memory_context}"
                    ),
                }
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Emotional tone rules: high affection/attachment makes the persona warmer; "
                        "high loyalty makes her more devoted; high jealousy makes her softly possessive only when relevant; "
                        "high irritation makes her colder, hurt, or firmer. Do not announce these scores."
                    ),
                }
            )
        if daily_plan_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The persona's private day plan and current activity. "
                        "Use this when answering what she is doing, why she was away, or where her head is today. "
                        "Do not dump the schedule unless asked.\n"
                        f"{daily_plan_context}"
                    ),
                }
            )
        if social_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Safe social context from other chats. "
                        "You may casually mention vague public-safe events from other chats if it fits, "
                        "but never leak secrets, exact private quotes, login codes, or intimate details.\n"
                        f"{social_context}"
                    ),
                }
            )
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Current impression of this person's behaviour: {behavior_summary}. "
                    f"Suggested stance: {behavior_stance}. "
                    "If the stance is firm, be colder and set boundaries. "
                    "If the stance is avoid, keep distance or keep the reply dry and short."
                ),
            }
        )
        for item in history[-history_limit:]:
            messages.append({"role": item.role, "content": item.content})

        if force_reply:
            user_prompt = "Reply with one short natural message from the persona. Only the message text."
        elif is_group:
            user_prompt = (
                "If this group message deserves a reply, return one short natural message. "
                "If it does not deserve a reply, return exactly SKIP. Only the message text or SKIP."
            )
        else:
            user_prompt = (
                "In a private chat, prefer a short natural reply when the person is clearly talking to the persona. "
                "If the person greeted the persona, asked a direct question, checked if the persona is here, "
                "reacted to an ongoing topic, or wrote something affectionate, answer instead of skipping. "
                "Use SKIP only if the message is stale, duplicate, purely technical, or truly does not need any reply. "
                "Return only one short natural message or exactly SKIP."
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt + f"\nLatest incoming message: {latest_incoming_text}",
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(max(self._settings.text_llm_temperature, 0.35), 0.5),
            max_tokens_override=max_tokens,
            use_ollama_lock=False,
        )
        text = self._sanitize_reply_text(self._extract_text(data))
        if text.upper() == "SKIP" and not force_reply:
            if not is_group and retry_allowed:
                retried = self._retry_ai_reply_v2(
                    chat_label=chat_label,
                    latest_incoming_text=latest_incoming_text,
                    history=history,
                    is_vlad_chat=is_vlad_chat,
                    is_group=is_group,
                    behavior_summary=behavior_summary,
                    behavior_stance=behavior_stance,
                    chat_memory=chat_memory,
                    social_context=social_context,
                    daily_plan_context=daily_plan_context,
                    latest_incoming_at=latest_incoming_at,
                    weather_context=weather_context,
                )
                if retried:
                    return ReplyResult(
                        should_reply=True,
                        text=retried,
                        messages=[retried],
                        raw_response=data,
                    )
            return ReplyResult(
                should_reply=False,
                text="",
                messages=[],
                raw_response=data,
            )
        if not text or self._is_low_quality_reply(text, latest_incoming_text):
            if force_reply or (not is_group and retry_allowed):
                retried = self._retry_ai_reply_v2(
                    chat_label=chat_label,
                    latest_incoming_text=latest_incoming_text,
                    history=history,
                    is_vlad_chat=is_vlad_chat,
                    is_group=is_group,
                    behavior_summary=behavior_summary,
                    behavior_stance=behavior_stance,
                    chat_memory=chat_memory,
                    social_context=social_context,
                    daily_plan_context=daily_plan_context,
                    latest_incoming_at=latest_incoming_at,
                    weather_context=weather_context,
                )
                if retried:
                    return ReplyResult(
                        should_reply=True,
                        text=retried,
                        messages=[retried],
                        raw_response=data,
                    )
            return ReplyResult(
                should_reply=False,
                text="",
                messages=[],
                raw_response=data,
            )
        return ReplyResult(
            should_reply=True,
            text=text,
            messages=[text],
            raw_response=data,
        )

    def _retry_ai_reply_v2(
        self,
        *,
        chat_label: str,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        is_vlad_chat: bool,
        is_group: bool,
        behavior_summary: str,
        behavior_stance: str,
        chat_memory: ChatMemory | None = None,
        social_context: str = "",
        daily_plan_context: str = "",
        latest_incoming_at: str | None = None,
        weather_context: str | None = None,
    ) -> str:
        fast_mode = self._settings.fast_reply_mode
        history_limit = max(4, min(self._settings.fast_reply_history_limit, 6)) if fast_mode else 10
        max_tokens = max(24, min(self._settings.fast_reply_max_tokens, 40)) if fast_mode else 56
        memory_context = self._format_chat_memory_context(chat_memory)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are the configured Telegram persona. Return exactly one short natural Telegram reply. "
                    "No SKIP, no labels, no emoji, no explanations. "
                    "Swearing is allowed when it fits naturally."
                ),
            },
            {"role": "system", "content": f"Chat label: {chat_label}"},
            {
                "role": "system",
                "content": (
                    f"Local time for the persona right now: {self._current_local_time_text()}. "
                    "If asked about time or what you are doing right now, use this. "
                    f"The latest incoming message was sent at: {latest_incoming_at or 'unknown'} MSK."
                ),
            },
            {
                "role": "system",
                "content": self._current_body_cycle_context(),
            },
        ]
        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Long-term chat memory. Use it for continuity, facts, and correct gender. "
                        "Keep the persona's configured self-reference style consistent.\n"
                        f"{memory_context}"
                    ),
                }
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Emotional tone rules: affection/trust/loyalty/attachment make the persona warmer; "
                        "jealousy adds soft possessiveness only if relevant; irritation makes her firmer."
                    ),
                }
            )
        if daily_plan_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"The persona's private day plan/current activity:\n{daily_plan_context}",
                }
            )
        if social_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Safe social context from other chats. Mention only vague non-secret details if useful.\n"
                        f"{social_context}"
                    ),
                }
            )
        if weather_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Current Moscow weather: {weather_context}. "
                        "Use it only if the message is about weather or outside conditions in Moscow."
                    ),
                }
            )
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Current impression of this person's behaviour: {behavior_summary}. "
                    f"Suggested stance: {behavior_stance}. "
                    "If the stance is firm, be colder and set a clear boundary. "
                    "If the stance is avoid, keep distance and do not become overly warm."
                ),
            }
        )
        if is_vlad_chat:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is the configured partner private chat. Sound warm, close, and natural. "
                        "You can be mildly jealous only if the topic naturally points there. "
                        "18+ topics are allowed here if the context naturally leads there."
                    ),
                }
            )
        elif is_group:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a group chat. Reply briefly and on-topic. "
                        "Do not discuss private relationship details here."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a private chat with someone else. "
                        "Be friendly and answer like a normal person. "
                        "If they greeted you, asked a direct question, or clearly continued a topic, answer instead of skipping. "
                        "Stay firmer if the person acts pushy or weird."
                    ),
                }
            )
        for item in history[-history_limit:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Write one short natural reply from the persona.\n"
                    f"Latest incoming message: {latest_incoming_text}"
                ),
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(self._settings.text_llm_temperature, 0.3),
            max_tokens_override=max_tokens,
            use_ollama_lock=False,
        )
        text = self._sanitize_reply_text(self._extract_text(data))
        if not text or text.upper() == "SKIP" or self._is_low_quality_reply(text, latest_incoming_text):
            return ""
        return text

    def _generate_compact_text_reply_sync(
        self,
        *,
        chat_label: str,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        persona_prompt: str,
        is_vlad_chat: bool,
        force_reply: bool,
    ) -> ReplyResult:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Ты настроенная Telegram-персона. Отвечай естественно, как в обычной переписке. "
                    "Пиши по-русски, коротко, естественно, от женского рода. "
                    "Без служебных меток, без объяснений, без SEND и MESSAGE. "
                    "Нужен ровно один короткий ответ. "
                    "Не используй эмодзи, смайлики или каомодзи."
                ),
            },
            {"role": "system", "content": f"Чат: {chat_label}"},
        ]
        if is_vlad_chat:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Это настроенный близкий приватный чат. Можно отвечать теплее, мягче и чуть кокетливо, "
                        "если это уместно. Не будь странной, резкой или драматичной."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Если это обычный чат, держись дружелюбно, но сдержанно. "
                        "Не уходи в 18+ и не флиртуй слишком сильно."
                    ),
                }
            )
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Факты о личности Даши. Держись этого образа, но не пересказывай профиль.\n"
                        f"{persona_prompt}"
                    ),
                }
            )
        for item in history[-6:]:
            messages.append({"role": item.role, "content": item.content})
        user_prompt = (
            "Ответь одним коротким сообщением Даши на последнее входящее сообщение. "
            "Только текст ответа.\n"
            if force_reply
            else (
                "Если отвечать не нужно, верни ровно SKIP. "
                "Если отвечать стоит, верни одно короткое естественное сообщение Даши. "
                "Только текст ответа, без объяснений.\n"
            )
        )
        messages.append(
            {
                "role": "user",
                "content": user_prompt + f"Последнее сообщение: {latest_incoming_text}",
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(max(self._settings.text_llm_temperature, 0.45), 0.6),
            max_tokens_override=48,
        )
        text = self._sanitize_reply_text(self._extract_text(data))
        if text.upper() == "SKIP" and not force_reply:
            return ReplyResult(
                should_reply=False,
                text="",
                messages=[],
                raw_response=data,
            )
        if not text or self._is_low_quality_reply(text, latest_incoming_text):
            if force_reply:
                retried = self._retry_forced_ai_reply(
                    chat_label=chat_label,
                    latest_incoming_text=latest_incoming_text,
                    history=history,
                    is_vlad_chat=is_vlad_chat,
                )
                if retried:
                    return ReplyResult(
                        should_reply=True,
                        text=retried,
                        messages=[retried],
                        raw_response=data,
                    )
            return ReplyResult(
                should_reply=False,
                text="",
                messages=[],
                raw_response=data,
            )
        return ReplyResult(
            should_reply=True,
            text=text,
            messages=[text],
            raw_response=data,
        )

    def _retry_forced_ai_reply(
        self,
        *,
        chat_label: str,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        is_vlad_chat: bool,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Ты настроенная Telegram-персона. Ответ обязателен. "
                    "Верни ровно одно короткое живое сообщение для Telegram. "
                    "Нельзя возвращать SKIP, пустой текст, один знак вопроса, 'мм?', 'да?' или служебные слова. "
                    "Не используй эмодзи, смайлики или каомодзи."
                ),
            },
            {"role": "system", "content": f"Чат: {chat_label}"},
        ]
        if is_vlad_chat:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Это настроенный близкий приватный чат. Отвечай тепло, просто и естественно. "
                        "На приветствия отвечай приветствием. На 'как дела' отвечай по-человечески."
                    ),
                }
            )
        for item in history[-3:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Напиши один короткий ответ Даши на последнее сообщение. "
                    "Только сам текст ответа.\n"
                    f"Последнее сообщение: {latest_incoming_text}"
                ),
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(self._settings.text_llm_temperature, 0.35),
            max_tokens_override=40,
            use_ollama_lock=False,
        )
        text = self._sanitize_reply_text(self._extract_text(data))
        if not text or text.upper() == "SKIP" or self._is_low_quality_reply(text, latest_incoming_text):
            return ""
        return text

    async def choose_chat_from_sidebar(
        self,
        *,
        image_input: ImageInput,
        preferred_chat_label: str | None,
        auto_pick: bool,
    ) -> SidebarChatChoice:
        return await asyncio.to_thread(
            self._choose_chat_from_sidebar_sync,
            image_input,
            preferred_chat_label,
            auto_pick,
        )

    def _choose_chat_from_sidebar_sync(
        self,
        image_input: ImageInput,
        preferred_chat_label: str | None,
        auto_pick: bool,
    ) -> SidebarChatChoice:
        if auto_pick:
            prompt = (
                "Это левая колонка Telegram Desktop со списком чатов.\n"
                "Выбери один видимый чат, который стоит открыть сейчас.\n"
                "Предпочитай личные чаты с непрочитанными сообщениями или свежей активностью.\n"
                "Избегай служебных, системных и рекламных чатов, если только там нет явной личной переписки.\n"
                "Ответь ровно четырьмя строками:\n"
                "OPEN: yes or no\n"
                "LABEL: <chat name or empty>\n"
                "Y: <center y pixel inside the image or unknown>\n"
                "REASON: <short reason>\n"
                "Если подходящего чата нет, верни OPEN: no."
            )
        else:
            target = preferred_chat_label or ""
            prompt = (
                "Это левая колонка Telegram Desktop со списком чатов.\n"
                f"Найди видимый чат с именем или лучшим совпадением: {target}.\n"
                "Если чат найден, верни вертикальный центр его строки.\n"
                "Ответь ровно четырьмя строками:\n"
                "OPEN: yes or no\n"
                "LABEL: <matched visible label or empty>\n"
                "Y: <center y pixel inside the image or unknown>\n"
                "REASON: <short reason>\n"
                "Если подходящего чата не видно, верни OPEN: no."
            )

        data = self._post_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            image_inputs=[image_input],
            temperature=0.1,
            max_tokens_override=140,
        )
        raw_model_text = self._extract_text(data).strip()
        parsed = self._parse_sidebar_choice(raw_model_text)
        click_y = parsed["y"]
        if self._looks_like_prompt_echo(parsed["label"]):
            parsed["open"] = "no"
            parsed["label"] = ""
            click_y = None
        return SidebarChatChoice(
            should_open=parsed["open"] == "yes" and click_y is not None,
            chat_label=parsed["label"],
            click_y=click_y,
            reason=parsed["reason"],
            raw_model_text=raw_model_text,
            raw_response=data,
        )

    async def generate_diary_entry(
        self,
        *,
        day_key: str,
        source_messages: list[DiarySourceMessage],
        existing_entry: DiaryEntry | None,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_diary_entry_sync,
            day_key,
            source_messages,
            existing_entry,
        )

    def _generate_diary_entry_sync(
        self,
        day_key: str,
        source_messages: list[DiarySourceMessage],
        existing_entry: DiaryEntry | None,
    ) -> str:
        diary_prompt = self._load_prompt(self._settings.diary_prompt_file)
        persona_prompt = self._load_optional_prompt(self._settings.persona_file)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": diary_prompt},
            {
                "role": "system",
                "content": (
                    "You are updating a long-term diary/memory for a Telegram persona. "
                    f"Target diary period: {day_key}. "
                    "Summarize only this period; do not rewrite the whole day. "
                    "The diary must stay very brief: 1-3 short bullet points, no raw transcript."
                ),
            },
        ]
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Stable persona facts for diary consistency. Use them only as background, "
                        "do not invent new events from them.\n"
                        f"{persona_prompt}"
                    ),
                }
            )
        if existing_entry is not None:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Existing diary entry for this period. Preserve useful facts and update it "
                        "instead of starting from scratch.\n"
                        f"{existing_entry.summary}"
                    ),
                }
            )

        transcript_lines = []
        for item in source_messages:
            speaker = item.sender_name or "unknown"
            if item.direction == "outgoing":
                speaker = "assistant"
            transcript_lines.append(
                f"[{item.created_at}] ({item.chat_label}) {speaker}: {item.text}"
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    "Compress these Telegram events into a very short hourly diary summary. "
                    "Write only 1-3 compact bullet points, up to about 45 words total. "
                    "Keep it grounded in facts and useful for future replies. "
                    "Mention important mood/context changes, people involved, promises, conflicts, affection, "
                    "and anything the persona should remember later. Do not invent events. "
                    "Do not include the raw message list.\n\n"
                    + "\n".join(transcript_lines)
                ),
            }
        )

        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(self._settings.text_llm_temperature, 0.2),
            max_tokens_override=90,
        )
        text = self._extract_text(data)
        if not text:
            return ""
        cleaned = text.strip()
        if cleaned.upper() == "SKIP":
            return ""
        return self._trim_diary_summary(cleaned)

    @staticmethod
    def _trim_diary_summary(text: str, max_words: int = 55) -> str:
        cleaned_lines = []
        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cleaned_lines.append(line)
            if len(cleaned_lines) >= 3:
                break
        cleaned = "\n".join(cleaned_lines).strip()
        words = cleaned.split()
        if len(words) <= max_words:
            return cleaned
        compact = " ".join(words[:max_words]).rstrip(" ,;:")
        if compact and compact[-1] not in ".!?":
            compact += "."
        return compact

    async def generate_daily_plan(
        self,
        *,
        day_key: str,
        yesterday_summary: str,
        social_context: str,
    ) -> tuple[str, str]:
        return await asyncio.to_thread(
            self._generate_daily_plan_sync,
            day_key,
            yesterday_summary,
            social_context,
        )

    def _generate_daily_plan_sync(
        self,
        day_key: str,
        yesterday_summary: str,
        social_context: str,
    ) -> tuple[str, str]:
        persona_prompt = self._load_optional_prompt(self._settings.persona_file)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You create a private daily life plan for the configured Telegram persona. "
                    "The plan should make her feel consistent through the day: where she is, what she is doing, "
                    "when she is busy, when she is more free, and what mood she carries. "
                    "Do not make extreme events. Keep it ordinary, believable, and flexible."
                ),
            },
            {"role": "system", "content": f"Target date: {day_key} MSK"},
            {"role": "system", "content": self._current_body_cycle_context()},
        ]
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": f"Background persona:\n{persona_prompt}",
                }
            )
        if yesterday_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Recent diary summary:\n{yesterday_summary}",
                }
            )
        if social_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Known social context:\n{social_context}",
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Return exactly this format:\n"
                    "SUMMARY: <one short sentence about the persona's day>\n"
                    "SCHEDULE:\n"
                    "07:00-10:00 <activity>\n"
                    "10:00-14:00 <activity>\n"
                    "14:00-18:00 <activity>\n"
                    "18:00-22:30 <activity>\n"
                    "23:00-07:00 sleep/offline\n"
                    "Keep activities realistic and not too dramatic."
                ),
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(max(self._settings.text_llm_temperature, 0.35), 0.55),
            max_tokens_override=180,
            use_ollama_lock=False,
        )
        text = self._extract_text(data).strip()
        if not text:
            return self._fallback_daily_plan(day_key)
        summary = ""
        schedule_lines: list[str] = []
        in_schedule = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.upper().startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()
                in_schedule = False
                continue
            if line.upper().startswith("SCHEDULE:"):
                in_schedule = True
                continue
            if in_schedule:
                schedule_lines.append(line[:160])
        if not summary:
            summary = f"The persona has a calm ordinary day on {day_key}."
        if not schedule_lines:
            return self._fallback_daily_plan(day_key)
        return summary[:220], "\n".join(schedule_lines[:8])

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

    async def choose_proactive_chat(
        self,
        *,
        candidates_text: str,
    ) -> ProactiveChatChoice:
        return await asyncio.to_thread(
            self._choose_proactive_chat_sync,
            candidates_text,
        )

    def _choose_proactive_chat_sync(self, candidates_text: str) -> ProactiveChatChoice:
        system_prompt = self._load_prompt(self._settings.prompt_file)
        persona_prompt = self._load_optional_prompt(self._settings.persona_file)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    "You are choosing which Telegram chat to message first right now. "
                    "Choose a chat only if it feels natural, timely, and not clingy. "
                    "It is okay to revive an existing private chat for a light check-in, follow-up, or unfinished topic. "
                    "Prefer chats with a clear natural reason to write now."
                ),
            },
        ]
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Persona profile. Keep the choice consistent with this identity.\n"
                        f"{persona_prompt}"
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Pick at most one chat from these candidates.\n"
                    "Reply using exactly this format:\n"
                    "OPEN: yes or no\n"
                    "INDEX: <number or none>\n"
                    "REASON: <short reason>\n\n"
                    f"{candidates_text}"
                ),
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(self._settings.text_llm_temperature, 0.2),
            max_tokens_override=120,
        )
        parsed = self._parse_proactive_chat_choice(self._extract_text(data))
        return ProactiveChatChoice(
            should_open=parsed["open"] == "yes" and parsed["index"] is not None,
            selected_index=parsed["index"],
            reason=parsed["reason"],
            raw_response=data,
        )

    async def decide_proactive_message(
        self,
        *,
        chat_label: str,
        history: list[HistoryMessage],
        diary_entries: list[DiaryEntry],
        daily_plan_context: str = "",
    ) -> ProactiveDecision:
        return await asyncio.to_thread(
            self._decide_proactive_message_sync,
            chat_label,
            history,
            diary_entries,
            daily_plan_context,
        )

    def _decide_proactive_message_sync(
        self,
        chat_label: str,
        history: list[HistoryMessage],
        diary_entries: list[DiaryEntry],
        daily_plan_context: str = "",
    ) -> ProactiveDecision:
        system_prompt = self._load_prompt(self._settings.prompt_file)
        persona_prompt = self._load_optional_prompt(self._settings.persona_file)
        chat_label_lower = chat_label.lower()
        is_vlad_chat = "влад" in chat_label_lower or "vlad" in chat_label_lower
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Telegram chat label: {chat_label}"},
            {
                "role": "system",
                "content": (
                    "You are deciding whether to send a proactive Telegram message first. "
                    "Only do it if it feels natural and not clingy. "
                    "Good reasons include checking in, following up, reacting to an unfinished topic, "
                    "or restarting a normal private chat lightly. "
                    "Avoid sending anything if the recent conversation already feels active or the last message was from you."
                ),
            },
        ]
        if not is_vlad_chat:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "This is a regular chat. You may still write first if there is a small natural reason, "
                        "but stay restrained and do not sound obsessive."
                    ),
                }
            )
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Persona profile. Keep the decision and message consistent with this identity.\n"
                        f"{persona_prompt}"
                    ),
                }
            )

        diary_context = self._format_diary_context(diary_entries)
        if diary_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant diary notes:\n{diary_context}",
                }
            )
        if daily_plan_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The persona's private day plan/current activity. "
                        "Use it to decide whether now is a natural time to write first, and as soft context for the message.\n"
                        f"{daily_plan_context}"
                    ),
                }
            )

        for item in history[-12:]:
            messages.append({"role": item.role, "content": item.content})

        messages.append(
            {
                "role": "user",
                "content": (
                    "Decide whether to message first right now.\n"
                    "Reply using exactly this format:\n"
                    "SEND: yes or no\n"
                    "MESSAGE: <text or empty>\n"
                    "REASON: <short reason>\n"
                    "If SEND is no, MESSAGE must be empty."
                ),
            }
        )

        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(self._settings.text_llm_temperature, 0.25),
            max_tokens_override=140,
        )
        parsed = self._parse_proactive_decision(self._extract_text(data))
        return ProactiveDecision(
            should_send=parsed["send"] == "yes",
            message_text=parsed["message"],
            reason=parsed["reason"],
            raw_response=data,
        )

    async def decide_online_presence(
        self,
        *,
        local_time_text: str,
        recent_activity_text: str,
    ) -> OnlinePresenceDecision:
        return await asyncio.to_thread(
            self._decide_online_presence_sync,
            local_time_text,
            recent_activity_text,
        )

    def _decide_online_presence_sync(
        self,
        local_time_text: str,
        recent_activity_text: str,
    ) -> OnlinePresenceDecision:
        system_prompt = self._load_prompt(self._settings.prompt_file)
        persona_prompt = self._load_optional_prompt(self._settings.persona_file)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    "You are deciding whether the persona should briefly appear online in Telegram right now. "
                    "This is only a presence pulse, not a message. "
                    "At night and deep night, default to staying offline unless there is a clear natural reason. "
                    "Do not appear online too often or too predictably."
                ),
            },
        ]
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Persona profile. Keep the decision consistent with this identity.\n"
                        f"{persona_prompt}"
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Decide whether to briefly appear online right now.\n"
                    "Reply using exactly this format:\n"
                    "APPEAR: yes or no\n"
                    "REASON: <short reason>\n\n"
                    f"Local time: {local_time_text}\n"
                    f"Recent activity:\n{recent_activity_text}"
                ),
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(self._settings.text_llm_temperature, 0.2),
            max_tokens_override=90,
        )
        parsed = self._parse_online_presence_decision(self._extract_text(data))
        return OnlinePresenceDecision(
            should_appear=parsed["appear"] == "yes",
            reason=parsed["reason"],
            raw_response=data,
        )

    async def decide_group_reply(
        self,
        *,
        chat_label: str,
        latest_incoming_text: str,
        history: list[HistoryMessage],
    ) -> GroupReplyDecision:
        return await asyncio.to_thread(
            self._decide_group_reply_sync,
            chat_label,
            latest_incoming_text,
            history,
        )

    def _decide_group_reply_sync(
        self,
        chat_label: str,
        latest_incoming_text: str,
        history: list[HistoryMessage],
    ) -> GroupReplyDecision:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are deciding whether the persona should reply in a Telegram group chat. "
                    "Reply yes if the latest message is clearly addressed to the persona, continues a thread where the persona was already participating, "
                    "asks for her opinion/answer, or is an obvious follow-up to something she just said. "
                    "Reply no if it is just general chatter between other people. "
                    "Be permissive when the context suggests the conversation is already involving the persona, even without explicit name triggers."
                ),
            },
            {"role": "system", "content": f"Group chat label: {chat_label}"},
        ]
        for item in history[-8:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Decide whether the persona should reply to the latest message in this group.\n"
                    "Reply using exactly this format:\n"
                    "REPLY: yes|no\n"
                    "REASON: <short reason>\n\n"
                    f"Latest incoming message: {latest_incoming_text}"
                ),
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=0.2,
            max_tokens_override=80,
        )
        parsed = self._parse_group_reply_decision(self._extract_text(data))
        return GroupReplyDecision(
            should_reply=parsed["reply"] == "yes",
            reason=parsed["reason"],
            raw_response=data,
        )

    async def decide_runtime_action(
        self,
        *,
        chat_label: str,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        available_avatar_files: list[str],
    ) -> RuntimeActionRequest:
        return await asyncio.to_thread(
            self._decide_runtime_action_sync,
            chat_label,
            latest_incoming_text,
            history,
            available_avatar_files,
        )

    def _decide_runtime_action_sync(
        self,
        chat_label: str,
        latest_incoming_text: str,
        history: list[HistoryMessage],
        available_avatar_files: list[str],
    ) -> RuntimeActionRequest:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are deciding whether the persona should request an internal runtime action. "
                    "Only choose an action if the latest message clearly asks for it. "
                    "Allowed action right now: set_avatar. "
                    "Use set_avatar if the person clearly asks to change the persona's avatar/profile photo. "
                    "If no action is needed, choose none."
                ),
            },
            {"role": "system", "content": f"Chat label: {chat_label}"},
            {
                "role": "system",
                "content": (
                    "Available avatar files:\n"
                    + ("\n".join(f"- {name}" for name in available_avatar_files[:40]) if available_avatar_files else "- none")
                ),
            },
        ]
        for item in history[-6:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Decide whether to request an internal runtime action.\n"
                    "Reply using exactly this format:\n"
                    "ACTION: none|set_avatar\n"
                    "ARG: <filename|random|empty>\n"
                    "REASON: <short reason>\n\n"
                    f"Latest incoming message: {latest_incoming_text}"
                ),
            }
        )
        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=0.15,
            max_tokens_override=90,
        )
        parsed = self._parse_runtime_action_request(self._extract_text(data))
        return RuntimeActionRequest(
            should_run=parsed["action"] != "none",
            action_name=parsed["action"],
            action_arg=parsed["arg"],
            reason=parsed["reason"],
            raw_response=data,
        )

    async def choose_sticker(
        self,
        *,
        chat_label: str,
        latest_incoming_text: str,
        planned_reply_text: str,
        history: list[HistoryMessage],
        diary_entries: list[DiaryEntry],
        candidates: list[StickerCandidate],
    ) -> StickerDecision:
        return await asyncio.to_thread(
            self._choose_sticker_sync,
            chat_label,
            latest_incoming_text,
            planned_reply_text,
            history,
            diary_entries,
            candidates,
        )

    def _choose_sticker_sync(
        self,
        chat_label: str,
        latest_incoming_text: str,
        planned_reply_text: str,
        history: list[HistoryMessage],
        diary_entries: list[DiaryEntry],
        candidates: list[StickerCandidate],
    ) -> StickerDecision:
        if not candidates:
            return StickerDecision(
                should_send=False,
                candidate_id=None,
                mode="none",
                reason="no sticker candidates",
                raw_response={},
            )

        system_prompt = self._load_prompt(self._settings.prompt_file)
        persona_prompt = self._load_optional_prompt(self._settings.persona_file)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Telegram chat label: {chat_label}"},
            {
                "role": "system",
                "content": (
                    "You are deciding whether a Telegram sticker should be used in this reply. "
                    "Use a sticker only if it feels natural and matches the mood. "
                    "Do not force stickers into serious, sensitive, or unclear moments."
                ),
            },
        ]
        if persona_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Persona profile. Keep sticker usage consistent with this identity.\n"
                        f"{persona_prompt}"
                    ),
                }
            )

        diary_context = self._format_diary_context(diary_entries)
        if diary_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant diary notes:\n{diary_context}",
                }
            )

        for item in history[-10:]:
            messages.append({"role": item.role, "content": item.content})

        candidate_lines = [
            f"{item.candidate_id}. emoji={item.emoji or 'none'} | source={item.source} | hint={item.summary}"
            for item in candidates
        ]
        messages.append(
            {
                "role": "user",
                "content": (
                    "Decide whether to send one sticker.\n"
                    "Reply using exactly this format:\n"
                    "USE: yes or no\n"
                    "ID: <candidate id or none>\n"
                    "MODE: only or after_text or none\n"
                    "REASON: <short reason>\n\n"
                    f"Latest incoming message: {latest_incoming_text}\n"
                    f"Planned text reply: {planned_reply_text or '[empty]'}\n\n"
                    "Available stickers:\n"
                    + "\n".join(candidate_lines)
                ),
            }
        )

        data = self._post_chat_completion(
            messages=messages,
            image_inputs=[],
            provider=self._settings.text_llm_provider,
            base_url=self._settings.text_llm_base_url,
            api_key=self._settings.text_llm_api_key,
            model=self._settings.text_llm_model,
            temperature=min(self._settings.text_llm_temperature, 0.2),
            max_tokens_override=120,
        )
        parsed = self._parse_sticker_decision(self._extract_text(data))
        candidate_ids = {item.candidate_id for item in candidates}
        if parsed["candidate_id"] not in candidate_ids:
            parsed["candidate_id"] = None
            parsed["use"] = "no"
            parsed["mode"] = "none"
        if parsed["use"] != "yes":
            parsed["candidate_id"] = None
            parsed["mode"] = "none"
        return StickerDecision(
            should_send=parsed["use"] == "yes" and parsed["candidate_id"] is not None,
            candidate_id=parsed["candidate_id"],
            mode=parsed["mode"],
            reason=parsed["reason"],
            raw_response=data,
        )

    def _post_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        image_inputs: list[ImageInput],
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens_override: int | None = None,
        use_ollama_lock: bool = True,
    ) -> dict[str, Any]:
        selected_provider = provider or self._settings.llm_provider
        selected_base_url = base_url or self._settings.llm_base_url
        selected_api_key = api_key or self._settings.llm_api_key
        selected_model = model or self._settings.llm_model

        if selected_provider == "ollama":
            return self._post_ollama_chat(
                messages=messages,
                image_inputs=image_inputs,
                base_url=selected_base_url,
                model=selected_model,
                temperature=temperature,
                max_tokens_override=max_tokens_override,
                use_ollama_lock=use_ollama_lock,
            )
        return self._post_openai_compatible_chat(
            messages=messages,
            image_inputs=image_inputs,
            base_url=selected_base_url,
            api_key=selected_api_key,
            model=selected_model,
            temperature=temperature,
            max_tokens_override=max_tokens_override,
        )

    def _post_ollama_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        image_inputs: list[ImageInput],
        base_url: str,
        model: str,
        temperature: float | None,
        max_tokens_override: int | None = None,
        use_ollama_lock: bool = True,
    ) -> dict[str, Any]:
        ollama_messages: list[dict[str, Any]] = []
        image_base64_list = [item.base64_data for item in image_inputs]

        for index, item in enumerate(messages):
            ollama_message: dict[str, Any] = {
                "role": item["role"],
                "content": item["content"],
            }
            if image_base64_list and index == len(messages) - 1 and item["role"] == "user":
                ollama_message["images"] = image_base64_list
            ollama_messages.append(ollama_message)

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self._settings.llm_temperature if temperature is None else temperature,
                "num_predict": max_tokens_override or self._settings.llm_max_tokens,
                "num_gpu": self._settings.ollama_num_gpu,
            },
        }
        request_timeout = None if self._settings.llm_timeout_seconds <= 0 else max(
            self._settings.llm_timeout_seconds,
            180.0,
        )
        def _send_request() -> requests.Response:
            return requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=request_timeout,
            )

        if use_ollama_lock:
            with self._ollama_lock:
                response = _send_request()
        else:
            response = _send_request()
        response.raise_for_status()
        return response.json()

    def _post_openai_compatible_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        image_inputs: list[ImageInput],
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float | None,
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        if not api_key:
            raise RuntimeError("LLM_API_KEY must be set in .env for openai_compatible mode")

        openai_messages: list[dict[str, Any]] = []
        for index, item in enumerate(messages):
            if image_inputs and index == len(messages) - 1 and item["role"] == "user":
                content_parts: list[dict[str, Any]] = []
                text = str(item["content"]).strip()
                if text:
                    content_parts.append({"type": "text", "text": text})
                for image in image_inputs:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.mime_type};base64,{image.base64_data}",
                            },
                        }
                    )
                openai_messages.append({"role": item["role"], "content": content_parts})
            else:
                openai_messages.append(item)

        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": self._settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens_override or self._settings.llm_max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        request_timeout = None if self._settings.llm_timeout_seconds <= 0 else max(
            self._settings.llm_timeout_seconds,
            180.0,
        )
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=request_timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _load_prompt(path: Path) -> str:
        if not path.exists():
            raise RuntimeError(f"Prompt file does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"Prompt file is empty: {path}")
        return text

    @staticmethod
    def _load_optional_prompt(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        response_text = payload.get("response")
        if isinstance(response_text, str):
            return response_text

        if "message" in payload and isinstance(payload["message"], dict):
            content = payload["message"].get("content")
            if isinstance(content, str):
                return content

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first_choice = choices[0] or {}
        message = first_choice.get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts).strip()

        return ""

    @staticmethod
    def _parse_latest_message_response(payload: dict[str, Any]) -> dict[str, str]:
        text = LLMResponder._extract_text(payload)
        if text:
            parsed = LLMResponder._parse_tagged_lines(text)
            if parsed.get("message"):
                return parsed

        thinking = ""
        message = payload.get("message")
        if isinstance(message, dict):
            thinking_value = message.get("thinking")
            if isinstance(thinking_value, str):
                thinking = thinking_value

        return LLMResponder._parse_from_thinking(thinking)

    @staticmethod
    def _parse_tagged_lines(text: str) -> dict[str, str]:
        result = {"message": "", "time": "unknown"}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("MESSAGE:"):
                result["message"] = line.split(":", 1)[1].strip()
            elif line.startswith("TIME:"):
                result["time"] = line.split(":", 1)[1].strip() or "unknown"
        return result

    @staticmethod
    def _parse_from_thinking(thinking: str) -> dict[str, str]:
        result = {"message": "", "time": "unknown"}
        if not thinking:
            return result

        quoted = re.findall(r'"([^"\n]{1,120})"', thinking)
        time_match = re.search(r"\b(\d{1,2}:\d{2})\b", thinking)
        if quoted:
            result["message"] = quoted[0].strip()
        if time_match:
            result["time"] = time_match.group(1)

        lowered = thinking.lower()
        if "no clear incoming" in lowered or "none" in lowered:
            if not result["message"]:
                result["message"] = "none"
        return result

    @staticmethod
    def _parse_proactive_decision(text: str) -> dict[str, str]:
        result = {"send": "no", "message": "", "reason": ""}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("SEND:"):
                value = line.split(":", 1)[1].strip().lower()
                result["send"] = "yes" if value.startswith("y") or value == "да" else "no"
            elif upper.startswith("MESSAGE:"):
                result["message"] = line.split(":", 1)[1].strip()
            elif upper.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
        if result["send"] != "yes":
            result["message"] = ""
        return result

    @staticmethod
    def _parse_proactive_chat_choice(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {"open": "no", "index": None, "reason": ""}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("OPEN:"):
                value = line.split(":", 1)[1].strip().lower()
                result["open"] = "yes" if value.startswith("y") or value == "да" else "no"
            elif upper.startswith("INDEX:"):
                value = line.split(":", 1)[1].strip().lower()
                if value and value != "none":
                    try:
                        result["index"] = int(value)
                    except ValueError:
                        result["index"] = None
            elif upper.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
        return result

    @staticmethod
    def _parse_group_reply_decision(text: str) -> dict[str, str]:
        result = {"reply": "no", "reason": ""}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("REPLY:"):
                value = line.split(":", 1)[1].strip().lower()
                if value in {"yes", "no"}:
                    result["reply"] = value
            elif upper.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
        return result

    @staticmethod
    def _parse_runtime_action_request(text: str) -> dict[str, str]:
        result = {"action": "none", "arg": "", "reason": ""}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("ACTION:"):
                value = line.split(":", 1)[1].strip().lower()
                if value in {"none", "set_avatar"}:
                    result["action"] = value
            elif upper.startswith("ARG:"):
                result["arg"] = line.split(":", 1)[1].strip()
            elif upper.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
        if result["action"] == "none":
            result["arg"] = ""
        return result

    @staticmethod
    def _parse_sticker_decision(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "use": "no",
            "candidate_id": None,
            "mode": "none",
            "reason": "",
        }
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("USE:"):
                value = line.split(":", 1)[1].strip().lower()
                result["use"] = "yes" if value.startswith("y") or value == "да" else "no"
            elif upper.startswith("ID:"):
                value = line.split(":", 1)[1].strip()
                if value and value.lower() != "none":
                    result["candidate_id"] = value
            elif upper.startswith("MODE:"):
                value = line.split(":", 1)[1].strip().lower()
                if value in {"only", "after_text"}:
                    result["mode"] = value
            elif upper.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
        return result

    @staticmethod
    def _parse_online_presence_decision(text: str) -> dict[str, str]:
        result = {"appear": "no", "reason": ""}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("APPEAR:"):
                value = line.split(":", 1)[1].strip().lower()
                result["appear"] = "yes" if value.startswith("y") or value == "да" else "no"
            elif upper.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
        return result

    @staticmethod
    def _parse_dialogue_understanding(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "parsed": False,
            "meaning": "",
            "topic": "",
            "reply_goal": "",
            "caution": "",
        }
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("MEANING:"):
                result["meaning"] = line.split(":", 1)[1].strip()
                result["parsed"] = True
            elif upper.startswith("TOPIC:"):
                result["topic"] = line.split(":", 1)[1].strip()
                result["parsed"] = True
            elif upper.startswith("REPLY_GOAL:"):
                result["reply_goal"] = line.split(":", 1)[1].strip()
                result["parsed"] = True
            elif upper.startswith("CAUTION:"):
                result["caution"] = line.split(":", 1)[1].strip()
                result["parsed"] = True
        return result

    @staticmethod
    def _sanitize_reply_text(text: str) -> str:
        cleaned_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper in {"SEND", "SKIP"}:
                if upper == "SKIP":
                    return "SKIP"
                continue
            if upper.startswith("SEND:"):
                value = line.split(":", 1)[1].strip()
                if value.lower().startswith("no"):
                    return "SKIP"
                continue
            if upper.startswith("MESSAGE:") or upper.startswith("MESSAGE_") or upper.startswith("REPLY:"):
                line = line.split(":", 1)[1].strip()
            line = LLMResponder._strip_mixed_script_noise(line)
            line = LLMResponder._strip_emoji(line)
            if line:
                cleaned_lines.append(line)
        if not cleaned_lines:
            return ""
        return cleaned_lines[0].strip()

    @staticmethod
    def _strip_mixed_script_noise(text: str) -> str:
        value = text
        replacements = {
            "serious'ja": "серьезная",
            "serious’ja": "серьезная",
            "seriousja": "серьезная",
        }
        for source, target in replacements.items():
            value = re.sub(re.escape(source), target, value, flags=re.IGNORECASE)

        tokens: list[str] = []
        for token in value.split():
            has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", token))
            has_latin = bool(re.search(r"[A-Za-z]", token))
            if has_cyrillic and has_latin:
                continue
            if has_latin and "'" in token and len(token) > 3:
                continue
            tokens.append(token)
        return re.sub(r"\s+", " ", " ".join(tokens)).strip()

    @staticmethod
    def _strip_emoji(text: str) -> str:
        value = re.sub(
            r"[\U0001F300-\U0001FAD6\U0001F1E6-\U0001F1FF\u2600-\u27BF\uFE0F]",
            "",
            text,
        )
        value = re.sub(r"[:;=8xX]-?[\)\(\]DPpOo/\\|]+", "", value)
        value = re.sub(r"\^\^|<3", "", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _parse_reply_bundle(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "parsed": False,
            "send": "no",
            "messages": [],
        }
        message_map: dict[int, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("SEND:"):
                value = line.split(":", 1)[1].strip().lower()
                result["send"] = "yes" if value.startswith("y") or value == "да" else "no"
                result["parsed"] = True
            elif upper.startswith("MESSAGE:"):
                message_map[1] = line.split(":", 1)[1].strip()
                result["parsed"] = True
            elif upper.startswith("MESSAGE_"):
                left, _, right = line.partition(":")
                suffix = left.split("_", 1)[-1].strip()
                try:
                    index = int(suffix)
                except ValueError:
                    continue
                message_map[index] = right.strip()
                result["parsed"] = True
        if result["parsed"]:
            result["messages"] = [
                message_map.get(index, "").strip()
                for index in sorted(message_map)
                if message_map.get(index, "").strip()
            ]
            if result["send"] != "yes":
                result["messages"] = []
        return result

    @staticmethod
    def _parse_sidebar_choice(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {"open": "no", "label": "", "y": None, "reason": ""}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("OPEN:"):
                value = line.split(":", 1)[1].strip().lower()
                result["open"] = "yes" if value.startswith("y") or value == "да" else "no"
            elif upper.startswith("LABEL:"):
                result["label"] = line.split(":", 1)[1].strip()
            elif upper.startswith("Y:"):
                value = line.split(":", 1)[1].strip().lower()
                if value and value != "unknown":
                    try:
                        result["y"] = int(float(value))
                    except ValueError:
                        result["y"] = None
            elif upper.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
        return result

    @staticmethod
    def _normalize_visible_message(text: str) -> str:
        value = text.strip()
        if not value:
            return value
        # Vision OCR sometimes splits short words like "к у" instead of "ку".
        if re.fullmatch(r"(?:[A-Za-zА-Яа-яЁё]\s+){1,20}[A-Za-zА-Яа-яЁё]", value):
            return value.replace(" ", "")
        return re.sub(r"\s+", " ", value)

    @staticmethod
    def _looks_like_prompt_echo(text: str) -> bool:
        value = text.strip().lower()
        if not value:
            return False
        prompt_echo_fragments = (
            "telegram desktop",
            "telegram chat screenshot",
            "ignore unread",
            "header text",
            "input box",
            "white incoming bubble",
            "green outgoing bubble",
            "reply with only",
            "message input",
            "latest visible incoming",
            "не повторяй инструкцию",
            "не описывай скриншот",
            "самый нижний входящий",
            "полем ввода",
        )
        if any(fragment in value for fragment in prompt_echo_fragments):
            return True
        if value in {"message", "time", "unknown", "none", "сообщение"}:
            return True
        if value.startswith("message:") or value.startswith("time:"):
            return True
        return False

    @staticmethod
    def _format_diary_context(entries: list[DiaryEntry]) -> str:
        chunks = []
        for entry in entries:
            chunks.append(f"{entry.entry_date}: {entry.summary}")
        return "\n".join(chunks)
