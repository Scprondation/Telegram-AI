# Telegram AI Assistant

[English version](README.en.md)

Telegram AI Assistant — это локальный Python-проект для Telegram-персоны, которая может отвечать в личных чатах и группах, учитывать историю переписки, вести компактный дневник памяти и работать с локальными или OpenAI-compatible LLM.

По умолчанию проект настроен безопасно для первого запуска: `AUTO_SEND=false`, проактивные сообщения выключены, чтение каналов выключено, приватные ключи не входят в репозиторий.

![Quickstart terminal](docs/screenshots/quickstart.png)

## Возможности

- Работа через Telegram API на базе `Telethon`.
- Опциональные режимы Telegram Desktop и Telegram Web через скриншоты.
- Генерация ответов через Ollama или OpenAI-compatible API.
- Отдельная текстовая модель и отдельная vision-модель для фото.
- Память диалогов в SQLite.
- Почасовой дневник в Markdown.
- Настраиваемые групповые триггеры.
- Опциональные стикеры, чтение каналов, проактивные сообщения и online-presence.
- Безопасный режим предпросмотра: при `AUTO_SEND=false` ответы пишутся в терминал, но не отправляются.

![Runtime overview](docs/screenshots/runtime-overview.png)

## Требования

- Python 3.11+.
- Telegram `api_id` и `api_hash` с [my.telegram.org](https://my.telegram.org/).
- Ollama, если используется локальная модель.
- Telegram-аккаунт, на котором вы готовы запускать user API сессию.

## Быстрый старт

1. Создайте окружение:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Установите зависимости для API-режима:

```powershell
pip install -r requirements-api.txt
```

Для desktop/web режимов установите полный набор:

```powershell
pip install -r requirements.txt
```

3. Установите Ollama и скачайте модели:

```powershell
ollama pull qwen3:8b
ollama pull qwen3-vl:8b
```

4. Создайте локальный конфиг:

```powershell
Copy-Item .env.example .env
```

5. Заполните в `.env`:

```env
TELEGRAM_API_ID=
TELEGRAM_API_HASH=
TELEGRAM_PHONE=
```

6. Запустите:

```powershell
python main.py
```

При первом запуске Telethon попросит код входа и создаст локальный `.session` файл. Никогда не выкладывайте `.env` и `.session` на GitHub.

## Основные настройки

```env
CLIENT_MODE=api
AUTO_SEND=false
TEXT_LLM_MODEL=qwen3:8b
LLM_MODEL=qwen3-vl:8b
VISION_ENABLED=true
DIARY_ENABLED=true
GROUP_REPLY_TRIGGERS=
```

`CLIENT_MODE=api` — рекомендуемый режим. Он не зависит от размера окна, OCR и состояния Telegram UI.

`AUTO_SEND=false` — безопасный тестовый режим. Включайте `AUTO_SEND=true` только после проверки поведения.

`GROUP_REPLY_TRIGGERS` — список слов, username или телефонов через запятую, по которым персона может отвечать в группах.

`PARTNER_CHAT_IDS` и `PARTNER_CHAT_NAMES` — опциональный список личных чатов, где персона должна отвечать теплее и стабильнее.

## Промты

- `prompt.txt` — основной стиль ответов.
- `persona.txt` — стабильный фон персонажа.
- `diary_prompt.txt` — правила сжатия событий в дневник.

Дефолтные промты обезличены. Перед реальным использованием настройте их под своего персонажа.

## Структура

```text
app/
  api_runtime.py       Telegram API runtime
  config.py            .env loader
  desktop_runtime.py   Telegram Desktop screenshot runtime
  responder.py         LLM client and prompt pipeline
  storage.py           SQLite memory store
  telegram_runtime.py  mode switch
deploy/
  telegram-ai-assistant.service  example systemd service
docs/screenshots/      README images
main.py                entrypoint
```

## Безопасность

- Не коммитьте `.env`, `.session`, `data/`, `gallery/`, `backups/`.
- Начинайте с `AUTO_SEND=false`.
- Проверяйте ответы перед включением автоотправки.
- User API автоматизация может нарушать правила Telegram при агрессивном использовании. Не спамьте и не запускайте проект на чужих аккаунтах.

## VPS

На Linux/VPS используйте API-режим:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
cp .env.example .env
python main.py
```

Для автозапуска можно адаптировать unit из `deploy/telegram-ai-assistant.service`.
