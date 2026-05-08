from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv


def _parse_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _parse_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _parse_csv(name: str) -> set[str]:
    value = os.getenv(name, "")
    items = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item:
            items.append(item.lower())
    return set(items)


def _parse_int_csv(name: str) -> set[int]:
    values: set[int] = set()
    for item in _parse_csv(name):
        try:
            values.add(int(item))
        except ValueError:
            continue
    return values


@dataclass(slots=True)
class Settings:
    project_root: Path
    client_mode: str
    telegram_api_id: int | None
    telegram_api_hash: str | None
    telegram_phone: str | None
    telegram_session_path: Path
    telegram_proxy_type: str | None
    telegram_proxy_host: str | None
    telegram_proxy_port: int | None
    telegram_proxy_username: str | None
    telegram_proxy_password: str | None
    telegram_proxy_secret: str | None
    telegram_web_url: str
    browser_user_data_dir: Path
    browser_channel: str | None
    browser_headless: bool
    desktop_process_names: set[str]
    desktop_chat_label: str | None
    desktop_auto_pick_chat: bool
    desktop_executable_path: Path | None
    desktop_auto_launch: bool
    poll_interval_seconds: float
    reply_debounce_seconds: float
    fast_reply_mode: bool
    fast_reply_skip_stickers: bool
    fast_reply_skip_retry: bool
    fast_reply_history_limit: int
    fast_reply_max_tokens: int
    fast_reply_background_cooldown_seconds: int
    login_wait_timeout_seconds: float
    skip_if_input_not_empty: bool
    llm_provider: str
    llm_base_url: str
    llm_api_key: str | None
    llm_model: str
    ollama_num_gpu: int
    text_llm_provider: str
    text_llm_base_url: str
    text_llm_api_key: str | None
    text_llm_model: str
    text_llm_temperature: float
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: float
    prompt_file: Path
    persona_file: Path
    diary_prompt_file: Path
    history_db_path: Path
    debug_dir: Path
    diary_markdown_dir: Path
    gallery_dir: Path
    chat_history_limit: int
    auto_send: bool
    trigger_prefix: str | None
    allowed_chats: set[str]
    group_reply_triggers: set[str]
    partner_chat_ids: set[int]
    partner_chat_names: set[str]
    proactive_enabled: bool
    proactive_allowed_chats: set[str]
    proactive_check_interval_seconds: int
    proactive_idle_minutes: int
    proactive_cooldown_minutes: int
    channels_auto_read: bool
    channel_history_limit: int
    online_presence_enabled: bool
    online_presence_check_interval_seconds: int
    online_presence_cooldown_minutes: int
    online_presence_min_visible_seconds: int
    online_presence_max_visible_seconds: int
    diary_enabled: bool
    diary_check_interval_seconds: int
    diary_min_messages_for_entry: int
    diary_min_new_messages: int
    diary_source_limit: int
    diary_lookup_limit: int
    diary_lookback_days: int
    stickers_enabled: bool
    sticker_candidate_limit: int
    sticker_cache_ttl_minutes: int
    vision_enabled: bool
    max_images_per_message: int
    debug_screenshots_enabled: bool


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if llm_provider not in {"ollama", "openai_compatible"}:
        raise RuntimeError("LLM_PROVIDER must be either 'ollama' or 'openai_compatible'")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    llm_api_key = os.getenv("LLM_API_KEY", "").strip() or None
    llm_model = os.getenv("LLM_MODEL", "").strip() or "qwen3-vl:4b"

    text_llm_provider = os.getenv("TEXT_LLM_PROVIDER", llm_provider).strip().lower()
    if text_llm_provider not in {"ollama", "openai_compatible"}:
        raise RuntimeError(
            "TEXT_LLM_PROVIDER must be either 'ollama' or 'openai_compatible'"
        )
    text_llm_base_url = os.getenv("TEXT_LLM_BASE_URL", llm_base_url).rstrip("/")
    text_llm_api_key = os.getenv("TEXT_LLM_API_KEY", "").strip() or llm_api_key
    text_llm_model = os.getenv("TEXT_LLM_MODEL", "").strip()
    if not text_llm_model:
        text_llm_model = "llama3.2:3b" if text_llm_provider == "ollama" else llm_model

    client_mode = os.getenv("CLIENT_MODE", "desktop").strip().lower()
    if client_mode not in {"desktop", "web", "api"}:
        raise RuntimeError("CLIENT_MODE must be either 'desktop', 'web', or 'api'")

    telegram_api_id_raw = os.getenv("TELEGRAM_API_ID", "").strip()
    telegram_api_id = int(telegram_api_id_raw) if telegram_api_id_raw else None
    telegram_api_hash = os.getenv("TELEGRAM_API_HASH", "").strip() or None
    telegram_phone = os.getenv("TELEGRAM_PHONE", "").strip() or None
    telegram_session_path = Path(os.getenv("TELEGRAM_SESSION_PATH", "assistant_user"))
    if not telegram_session_path.is_absolute():
        telegram_session_path = project_root / telegram_session_path

    telegram_proxy_type = os.getenv("TELEGRAM_PROXY_TYPE", "").strip().lower() or None
    telegram_proxy_host = os.getenv("TELEGRAM_PROXY_HOST", "").strip() or None
    telegram_proxy_port_raw = os.getenv("TELEGRAM_PROXY_PORT", "").strip()
    telegram_proxy_port = int(telegram_proxy_port_raw) if telegram_proxy_port_raw else None
    telegram_proxy_username = os.getenv("TELEGRAM_PROXY_USERNAME", "").strip() or None
    telegram_proxy_password = os.getenv("TELEGRAM_PROXY_PASSWORD", "").strip() or None
    telegram_proxy_secret = os.getenv("TELEGRAM_PROXY_SECRET", "").strip() or None
    telegram_proxy_url = os.getenv("TELEGRAM_PROXY_URL", "").strip()
    if telegram_proxy_url:
        parsed_proxy = _parse_telegram_proxy_url(telegram_proxy_url)
        telegram_proxy_type = parsed_proxy["type"] or telegram_proxy_type
        telegram_proxy_host = parsed_proxy["host"] or telegram_proxy_host
        telegram_proxy_port = parsed_proxy["port"] or telegram_proxy_port
        telegram_proxy_secret = parsed_proxy["secret"] or telegram_proxy_secret

    prompt_file = Path(os.getenv("PROMPT_FILE", "prompt.txt"))
    if not prompt_file.is_absolute():
        prompt_file = project_root / prompt_file

    diary_prompt_file = Path(os.getenv("DIARY_PROMPT_FILE", "diary_prompt.txt"))
    if not diary_prompt_file.is_absolute():
        diary_prompt_file = project_root / diary_prompt_file

    persona_file = Path(os.getenv("PERSONA_FILE", "persona.txt"))
    if not persona_file.is_absolute():
        persona_file = project_root / persona_file

    history_db_path = Path(os.getenv("HISTORY_DB_PATH", "data/history.sqlite3"))
    if not history_db_path.is_absolute():
        history_db_path = project_root / history_db_path
    history_db_path.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(os.getenv("DEBUG_DIR", "data/debug"))
    if not debug_dir.is_absolute():
        debug_dir = project_root / debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)

    diary_markdown_dir = Path(os.getenv("DIARY_MARKDOWN_DIR", "data/diary"))
    if not diary_markdown_dir.is_absolute():
        diary_markdown_dir = project_root / diary_markdown_dir
    diary_markdown_dir.mkdir(parents=True, exist_ok=True)

    gallery_dir = Path(os.getenv("GALLERY_DIR", "gallery"))
    if not gallery_dir.is_absolute():
        gallery_dir = project_root / gallery_dir
    gallery_dir.mkdir(parents=True, exist_ok=True)

    browser_user_data_dir = Path(
        os.getenv("BROWSER_USER_DATA_DIR", "data/telegram_web_profile")
    )
    if not browser_user_data_dir.is_absolute():
        browser_user_data_dir = project_root / browser_user_data_dir
    browser_user_data_dir.mkdir(parents=True, exist_ok=True)

    trigger_prefix = os.getenv("TRIGGER_PREFIX", "").strip()
    group_reply_triggers = _parse_csv("GROUP_REPLY_TRIGGERS")
    desktop_chat_label = os.getenv("DESKTOP_CHAT_LABEL", "").strip() or None
    desktop_executable_path_raw = os.getenv("DESKTOP_EXECUTABLE_PATH", "").strip()
    desktop_executable_path = Path(desktop_executable_path_raw) if desktop_executable_path_raw else None

    desktop_process_names = _parse_csv("DESKTOP_PROCESS_NAMES")
    if not desktop_process_names:
        desktop_process_names = {"ayugram", "telegram", "telegramdesktop"}

    return Settings(
        project_root=project_root,
        client_mode=client_mode,
        telegram_api_id=telegram_api_id,
        telegram_api_hash=telegram_api_hash,
        telegram_phone=telegram_phone,
        telegram_session_path=telegram_session_path,
        telegram_proxy_type=telegram_proxy_type,
        telegram_proxy_host=telegram_proxy_host,
        telegram_proxy_port=telegram_proxy_port,
        telegram_proxy_username=telegram_proxy_username,
        telegram_proxy_password=telegram_proxy_password,
        telegram_proxy_secret=telegram_proxy_secret,
        telegram_web_url=os.getenv("TELEGRAM_WEB_URL", "https://web.telegram.org/a/").strip(),
        browser_user_data_dir=browser_user_data_dir,
        browser_channel=os.getenv("BROWSER_CHANNEL", "chrome").strip() or None,
        browser_headless=_parse_bool("BROWSER_HEADLESS", False),
        desktop_process_names=desktop_process_names,
        desktop_chat_label=desktop_chat_label,
        desktop_auto_pick_chat=_parse_bool("DESKTOP_AUTO_PICK_CHAT", False),
        desktop_executable_path=desktop_executable_path,
        desktop_auto_launch=_parse_bool("DESKTOP_AUTO_LAUNCH", True),
        poll_interval_seconds=_parse_float("POLL_INTERVAL_SECONDS", 2.0),
        reply_debounce_seconds=_parse_float("REPLY_DEBOUNCE_SECONDS", 0.35),
        fast_reply_mode=_parse_bool("FAST_REPLY_MODE", True),
        fast_reply_skip_stickers=_parse_bool("FAST_REPLY_SKIP_STICKERS", True),
        fast_reply_skip_retry=_parse_bool("FAST_REPLY_SKIP_RETRY", True),
        fast_reply_history_limit=_parse_int("FAST_REPLY_HISTORY_LIMIT", 8),
        fast_reply_max_tokens=_parse_int("FAST_REPLY_MAX_TOKENS", 48),
        fast_reply_background_cooldown_seconds=_parse_int(
            "FAST_REPLY_BACKGROUND_COOLDOWN_SECONDS",
            75,
        ),
        login_wait_timeout_seconds=_parse_float("LOGIN_WAIT_TIMEOUT_SECONDS", 600.0),
        skip_if_input_not_empty=_parse_bool("SKIP_IF_INPUT_NOT_EMPTY", True),
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        ollama_num_gpu=_parse_int("OLLAMA_NUM_GPU", 0),
        text_llm_provider=text_llm_provider,
        text_llm_base_url=text_llm_base_url,
        text_llm_api_key=text_llm_api_key,
        text_llm_model=text_llm_model,
        text_llm_temperature=_parse_float("TEXT_LLM_TEMPERATURE", 0.35),
        llm_temperature=_parse_float("LLM_TEMPERATURE", 0.9),
        llm_max_tokens=_parse_int("LLM_MAX_TOKENS", 350),
        llm_timeout_seconds=_parse_float("LLM_TIMEOUT_SECONDS", 60.0),
        prompt_file=prompt_file,
        persona_file=persona_file,
        diary_prompt_file=diary_prompt_file,
        history_db_path=history_db_path,
        debug_dir=debug_dir,
        diary_markdown_dir=diary_markdown_dir,
        gallery_dir=gallery_dir,
        chat_history_limit=_parse_int("CHAT_HISTORY_LIMIT", 20),
        auto_send=_parse_bool("AUTO_SEND", False),
        trigger_prefix=trigger_prefix or None,
        allowed_chats=_parse_csv("ALLOWED_CHATS"),
        group_reply_triggers=group_reply_triggers,
        partner_chat_ids=_parse_int_csv("PARTNER_CHAT_IDS"),
        partner_chat_names=_parse_csv("PARTNER_CHAT_NAMES"),
        proactive_enabled=_parse_bool("PROACTIVE_ENABLED", True),
        proactive_allowed_chats=_parse_csv("PROACTIVE_ALLOWED_CHATS"),
        proactive_check_interval_seconds=_parse_int("PROACTIVE_CHECK_INTERVAL_SECONDS", 180),
        proactive_idle_minutes=_parse_int("PROACTIVE_IDLE_MINUTES", 20),
        proactive_cooldown_minutes=_parse_int("PROACTIVE_COOLDOWN_MINUTES", 90),
        channels_auto_read=_parse_bool("CHANNELS_AUTO_READ", True),
        channel_history_limit=_parse_int("CHANNEL_HISTORY_LIMIT", 12),
        online_presence_enabled=_parse_bool("ONLINE_PRESENCE_ENABLED", True),
        online_presence_check_interval_seconds=_parse_int("ONLINE_PRESENCE_CHECK_INTERVAL_SECONDS", 900),
        online_presence_cooldown_minutes=_parse_int("ONLINE_PRESENCE_COOLDOWN_MINUTES", 45),
        online_presence_min_visible_seconds=_parse_int("ONLINE_PRESENCE_MIN_VISIBLE_SECONDS", 35),
        online_presence_max_visible_seconds=_parse_int("ONLINE_PRESENCE_MAX_VISIBLE_SECONDS", 110),
        diary_enabled=_parse_bool("DIARY_ENABLED", True),
        diary_check_interval_seconds=_parse_int("DIARY_CHECK_INTERVAL_SECONDS", 3600),
        diary_min_messages_for_entry=_parse_int("DIARY_MIN_MESSAGES_FOR_ENTRY", 8),
        diary_min_new_messages=_parse_int("DIARY_MIN_NEW_MESSAGES", 4),
        diary_source_limit=_parse_int("DIARY_SOURCE_LIMIT", 200),
        diary_lookup_limit=_parse_int("DIARY_LOOKUP_LIMIT", 3),
        diary_lookback_days=_parse_int("DIARY_LOOKBACK_DAYS", 30),
        stickers_enabled=_parse_bool("STICKERS_ENABLED", True),
        sticker_candidate_limit=_parse_int("STICKER_CANDIDATE_LIMIT", 24),
        sticker_cache_ttl_minutes=_parse_int("STICKER_CACHE_TTL_MINUTES", 30),
        vision_enabled=_parse_bool("VISION_ENABLED", True),
        max_images_per_message=_parse_int("MAX_IMAGES_PER_MESSAGE", 1),
        debug_screenshots_enabled=_parse_bool("DEBUG_SCREENSHOTS_ENABLED", True),
    )


def _parse_telegram_proxy_url(url: str) -> dict[str, str | int | None]:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "tg"}:
        return {"type": None, "host": None, "port": None, "secret": None}
    query = parse_qs(parsed.query)
    host = (query.get("server") or [None])[0]
    port_raw = (query.get("port") or [None])[0]
    secret = (query.get("secret") or [None])[0]
    port = int(port_raw) if port_raw else None
    return {"type": "mtproto", "host": host, "port": port, "secret": secret}
