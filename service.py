import asyncio
import html
import json
import logging
import os
import re
import time
from collections.abc import Callable
from datetime import date

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from openai import (
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, RSS_FEEDS
from database import is_link_posted, save_link, save_post

# cross-platform audio dir
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Groq client ---

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
)

# --- Error messages ---

OVERLOAD_MSG = (
    "⚠️ Система перевантажена або ви надсилаєте запити занадто швидко. "
    "Будь ласка, зачекайте 1 хвилину."
)
LIMIT_REACHED_MSG = OVERLOAD_MSG  # backward compat
THROTTLE_MSG      = OVERLOAD_MSG
RATE_LIMIT_MSG    = OVERLOAD_MSG
BAN_429_MSG       = OVERLOAD_MSG

BAD_REQUEST_MSG  = "⚠️ Запит занадто складний або некоректний."
SERVER_ERROR_MSG = "⚠️ Помилка на стороні сервера Groq. Спробуй пізніше."

# --- Daily token monitor ---

DAILY_TOKEN_LIMIT = 100_000
ALERT_THRESHOLD   = 0.80

_daily_usage: dict = {"date": "", "tokens": 0, "alert_sent": False}
_bot_instance      = None


def set_bot_instance(bot) -> None:
    global _bot_instance
    _bot_instance = bot


async def check_usage_limits(tokens_used: int, admin_id: int) -> None:
    # reset on date change; alert once at >80% TPD
    today = date.today().isoformat()

    if _daily_usage["date"] != today:
        _daily_usage.update({"date": today, "tokens": 0, "alert_sent": False})

    _daily_usage["tokens"] += tokens_used
    used  = _daily_usage["tokens"]
    ratio = used / DAILY_TOKEN_LIMIT

    logging.debug(f"[token_monitor] {used}/{DAILY_TOKEN_LIMIT} ({ratio:.1%})")

    if ratio >= ALERT_THRESHOLD and not _daily_usage["alert_sent"] and _bot_instance:
        _daily_usage["alert_sent"] = True
        alert = (
            f"🚨 <b>Увага! Вичерпано 80% добового ліміту токенів.</b>\n\n"
            f"📊 Використано: <code>{used:,}</code> / "
            f"<code>{DAILY_TOKEN_LIMIT:,}</code> токенів\n"
            f"📈 Завантаження: <b>{ratio:.1%}</b>\n\n"
            f"💡 Ліміт скинеться опівночі (UTC)."
        )
        try:
            await _bot_instance.send_message(admin_id, alert, parse_mode="HTML")
            logging.warning(f"[token_monitor] Alert надіслано: {used} токенів")
        except Exception as e:
            logging.error(f"[token_monitor] Не вдалося надіслати alert: {e}")


# --- Formatting utils ---

_ERROR_PREFIXES = ("⚠️", "😔")


def _is_error_reply(text: str) -> bool:
    return any(text.startswith(p) for p in _ERROR_PREFIXES)


def _escape_and_wrap_code(text: str) -> str:
    # odd split parts → <code>
    parts  = re.split(r"```(?:\w+)?\n?(.*?)```", text, flags=re.DOTALL)
    result: list[str] = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            result.append(html.escape(part))
        else:
            result.append(f"<code>{html.escape(part.strip())}</code>")

    return "".join(result)


# --- Groq API wrapper ---

async def _groq_create(
    messages:      list,
    temperature:   float                       = 0.2,
    max_tokens:    int                         = 1500,
    json_mode:     bool                        = False,
    user_id:       int | None                  = None,
    admin_id:      int | None                  = None,
    on_rate_limit: Callable[[int], None] | None = None,
    tools:         list | None                 = None,
) -> str | dict:
    # 500 → retry once; 429/400 → no retry
    kwargs: dict = dict(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if tools:
        kwargs["tools"] = tools

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(**kwargs)

            if admin_id is not None and response.usage:
                await check_usage_limits(response.usage.total_tokens, admin_id)

            choice  = response.choices[0]
            message = choice.message

            if message.tool_calls:
                parsed_calls = []
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    parsed_calls.append({
                        "name":      tc.function.name,
                        "arguments": args,
                    })
                    logging.info(
                        f"[tool_call] 🛠 Модель обрала інструмент: "
                        f"'{tc.function.name}' · args={tc.function.arguments[:120]}"
                    )
                return {"tool_calls": parsed_calls}

            return message.content

        except RateLimitError:
            logging.warning(f"Groq RateLimitError (429) · user={user_id}")
            if user_id is not None and on_rate_limit is not None:
                on_rate_limit(user_id)  # triggers ban in chatmanager
            return RATE_LIMIT_MSG

        except BadRequestError as e:
            logging.error(f"Groq BadRequestError (400): {e}")
            return BAD_REQUEST_MSG

        except InternalServerError as e:
            logging.warning(f"Groq InternalServerError (500) attempt={attempt}: {e}")
            if attempt == 0:
                await asyncio.sleep(1)
                continue
            return SERVER_ERROR_MSG

        except Exception as e:
            logging.error(f"Groq unexpected error: {e}")
            return "😔 Щось пішло не так. Спробуй ще раз пізніше."

    return SERVER_ERROR_MSG


# --- Voice assistant ---

async def transcribe_and_answer(audio_path: str, user_id: int) -> str:
    import aiofiles
    from chatmanager import ask_groq  # avoid circular import

    try:
        async with aiofiles.open(audio_path, "rb") as f:
            audio_bytes = await f.read()

        transcription = await client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("voice.ogg", audio_bytes, "audio/ogg"),
            response_format="text",
        )

        if not transcription or not transcription.strip():
            return (
                "😔 Не вдалося розпізнати мовлення. "
                "Спробуй говорити чіткіше або ближче до мікрофона."
            )

        recognized_text = transcription.strip()
        logging.info(f"[whisper] user={user_id}: {recognized_text[:80]}")

        answer = await ask_groq(user_id, recognized_text)

        return (
            f"🎙 <b>Ти сказав:</b>\n"
            f"<i>{html.escape(recognized_text)}</i>\n\n"
            f"🤖 <b>Відповідь:</b>\n{answer}"
        )

    except RateLimitError:
        logging.warning("transcribe_and_answer: Groq RateLimitError (429)")
        return RATE_LIMIT_MSG

    except BadRequestError as e:
        logging.error(f"transcribe_and_answer: BadRequestError (400): {e}")
        return BAD_REQUEST_MSG

    except InternalServerError as e:
        logging.error(f"transcribe_and_answer: InternalServerError (500): {e}")
        return SERVER_ERROR_MSG

    except Exception as e:
        logging.error(f"transcribe_and_answer error: {e}")
        return "😔 Помилка при обробці голосового повідомлення. Спробуй ще раз."

    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass


# --- RSS pipeline system prompt (JSON mode) ---

SYSTEM_PROMPT = (
    "Ти — україномовний ШІ-помічник Mindflow AI. "
    "Твоя єдина мова спілкування — українська. "
    "Категорично заборонено використовувати будь-які інші мови. "
    "Проаналізуй текст новини та відповідай, використовуючи сучасну українську термінологію. "
    "Видай відповідь строго у форматі JSON:\n"
    "{\n"
    "  \"name\": \"заголовок новини одною фразою\",\n"
    "  \"desc\": \"суть новини — 1 речення українською\",\n"
    "  \"feature\": \"головна фішка — 1 коротка фраза\",\n"
    "  \"pros\": [\"перевага1\", \"перевага2\", \"перевага3\"],\n"
    "  \"prompt\": \"приклад корисного промпту українською\",\n"
    "  \"tool_name\": \"точна назва ШІ-інструменту або порожній рядок\",\n"
    "  \"tool_url\": \"пряме посилання на сайт або порожній рядок\"\n"
    "}"
)


# --- RSS parser ---

class RSSParser:
    async def fetch_new_articles(self) -> list:
        new_articles = []
        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    url = entry.get("link", "")
                    if url and not await is_link_posted(url):
                        new_articles.append({
                            "title":   entry.get("title", ""),
                            "url":     url,
                            "summary": entry.get("summary", ""),
                        })
                        logging.info(f"[rss] Нова стаття: {entry.get('title', '')}")
            except Exception as e:
                logging.error(f"[rss] Помилка парсингу {feed_url}: {e}")
        return new_articles

    async def fetch_article_text(self, url: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    raw_html = await resp.text()
                    soup = BeautifulSoup(raw_html, "html.parser")
                    text = " ".join(p.get_text() for p in soup.find_all("p"))
                    return text[:2000]
        except Exception as e:
            logging.error(f"[rss] Помилка завантаження {url}: {e}")
            return ""


# --- AI manager (Groq JSON mode) ---

class AIManager:
    async def process_article(self, title: str, text: str) -> dict | None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Заголовок: {title}\n\nТекст: {text}"},
        ]

        raw = await _groq_create(
            messages, temperature=0.2, max_tokens=1000, json_mode=True
        )

        if _is_error_reply(raw):
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error(f"[ai_manager] JSON parse error: {e}")
            return None


# --- Post formatter ---

def format_post(data: dict, url: str) -> str:
    pros = "\n".join(f"• {html.escape(p)}" for p in data.get("pros", []))
    return (
        f"🤖 <b>{html.escape(data.get('name', 'Нова нейромережа'))}</b>\n\n"
        f"📝 {html.escape(data.get('desc', ''))}\n\n"
        f"💼 <b>Чим корисна:</b>\n{pros}\n\n"
        f"💡 <b>Приклад промпту:</b>\n<i>{html.escape(data.get('prompt', ''))}</i>\n\n"
        f"🔗 <a href='{url}'>Читати повністю</a>"
    )


# --- Main pipeline: RSS → Groq → DB ---

async def run_pipeline() -> int:
    """Returns count of saved records."""
    parser = RSSParser()
    ai     = AIManager()

    articles    = await parser.fetch_new_articles()
    saved_total = 0

    for article in articles:
        text = await parser.fetch_article_text(article["url"])
        if not text:
            text = article["summary"]

        result = await ai.process_article(article["title"], text)
        if not result:
            continue

        try:
            await save_post(
                url=article["url"],
                name=result.get("name", ""),
                desc=result.get("desc", ""),
                pros=result.get("pros", []),
                prompt=result.get("prompt", ""),
                tool_name=result.get("tool_name", ""),
                tool_url=result.get("tool_url", ""),
            )
            await save_link(article["url"])
            logging.info(f"[pipeline] Збережено: {result.get('name', article['url'])}")
            saved_total += 1
        except Exception as e:
            logging.error(f"[pipeline] Помилка збереження {article['url']}: {e}")

    return saved_total