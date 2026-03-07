import asyncio
import html
import logging
import os
import time
import traceback
from datetime import datetime

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    ContentType,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import ADMIN_ID, BOT_TOKEN
from database import add_user, get_last_posts, init_db, search_news
from service import (
    OVERLOAD_MSG,
    TEMP_DIR,
    run_pipeline,
    set_bot_instance,
    transcribe_and_answer,
)
from chatmanager import (
    REJECT_INPUT_MSG,
    ask_groq,
    compare_ai,
    improve_prompt,
    validate_input,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())

set_bot_instance(bot)

_error_alert_cache: dict[str, float] = {}  # alert dedup cache
ALERT_COOLDOWN_SECONDS = 300               # 5 min cooldown


# --- Throttling middleware ---

class ThrottlingMiddleware(BaseMiddleware):
    THROTTLE_SECONDS = 2.0
    WARN_COOLDOWN    = 5.0

    def __init__(self):
        self._last_message: dict[int, float] = {}
        self._last_warning: dict[int, float] = {}

    async def __call__(self, handler, event: Message, data: dict):
        user_id = event.from_user.id if event.from_user else None

        if user_id is not None:
            now = time.monotonic()

            if now - self._last_message.get(user_id, 0) < self.THROTTLE_SECONDS:
                if now - self._last_warning.get(user_id, 0) >= self.WARN_COOLDOWN:
                    self._last_warning[user_id] = now
                    await event.answer(OVERLOAD_MSG)
                return None  # block regardless of warning

            self._last_message[user_id] = now

        return await handler(event, data)


# --- FSM states ---

class SearchState(StatesGroup):
    waiting_for_query = State()

class PromptState(StatesGroup):
    waiting_for_prompt = State()

class CompareState(StatesGroup):
    waiting_for_models = State()


# --- Global error handler ---

async def global_error_handler(event: ErrorEvent) -> bool:
    exception  = event.exception
    error_type = html.escape(type(exception).__name__)
    error_msg  = html.escape(str(exception)[:300])

    logging.exception(f"Необроблена помилка [{error_type}]: {exception}")

    now       = time.monotonic()
    last_sent = _error_alert_cache.get(error_type, 0)
    if now - last_sent < ALERT_COOLDOWN_SECONDS:
        return True
    _error_alert_cache[error_type] = now

    tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    short_tb  = "".join(tb_lines[-10:]).strip()
    if len(short_tb) > 900:
        short_tb = "..." + short_tb[-900:]
    short_tb = html.escape(short_tb)

    now_str    = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    alert_text = (
        f"🚨 <b>Критична помилка!</b>\n\n"
        f"🔴 Тип: <code>{error_type}</code>\n"
        f"🕒 Час: {now_str}\n\n"
        f"📄 Деталі: <code>{error_msg}</code>\n\n"
        f"<pre>{short_tb}</pre>"
    )

    try:
        await bot.send_message(ADMIN_ID, alert_text, parse_mode="HTML")
    except Exception as send_err:
        logging.error(f"Не вдалося надіслати алерт адміну: {send_err}")

    return True


# --- Keyboards ---

def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚡️ AI-Дайджест",        callback_data="last_post")],
        [InlineKeyboardButton(text="🔎 Пошук інструментів",  callback_data="search")],
        [InlineKeyboardButton(text="🪄 Покращити промпт",    callback_data="prompt_trainer")],
        [InlineKeyboardButton(text="📊 Порівняти моделі",    callback_data="compare_ai")],
        [InlineKeyboardButton(text="🛠 Замовити розробку",   callback_data="about")],
    ])


def cancel_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="❌ Скасувати")]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


# --- /start ---

@dp.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.clear()
    await add_user(message.from_user.id, message.from_user.username or "")
    await message.answer(
        "💠 <b>Mindflow AI</b>\n\n"
        "Автономний ШІ-агент, який стежить за світом нейромереж замість тебе.\n\n"
        "⚡️ Свіжі інструменти · 🔎 Розумний пошук · 🛠 Зроблено на Groq LPU",
        reply_markup=main_keyboard(),
        parse_mode="HTML",
    )


# --- About / order dev ---

@dp.callback_query(F.data == "about")
async def cb_about(callback: CallbackQuery):
    dev_username = "neverlookbacks"
    text = (
        "🛠 <b>Хочеш такого ж бота для свого бізнесу?</b>\n\n"
        "Цей проєкт — приклад того, як сучасний ШІ може повністю замінити "
        "контент-менеджера. Я спеціалізуюся на створенні <b>автономних ШІ-асистентів</b> "
        "на базі Llama 3.3 та Groq — систем, які ідеально розуміють український контекст.\n\n"
        "💼 <b>Що я можу реалізувати для вас:</b>\n"
        "• Парсери новин з автоматичним перекладом та підсумуванням\n"
        "• Інтеграцію розумних чат-консультантів (Groq / OpenAI) у ваш бізнес\n"
        "• Складні системи автоматизації з базами даних та адмін-панелями\n"
        "• Україномовні ШІ-боти для будь-якої ніші\n\n"
        "⚡️ Мої рішення працюють швидше завдяки <b>чіпам LPU від Groq</b>.\n\n"
        f"👨‍💻 Зв'язатися: <a href='https://t.me/{dev_username}'>@{dev_username}</a>"
    )
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💬 Написати розробнику", url=f"https://t.me/{dev_username}")]
    ])
    await callback.message.answer(
        text, parse_mode="HTML", reply_markup=keyboard, disable_web_page_preview=True,
    )
    await callback.answer()


# --- AI digest ---

@dp.callback_query(F.data == "last_post")
async def cb_last_post(callback: CallbackQuery):
    posts = await get_last_posts(limit=3)

    if not posts:
        await callback.message.answer(
            "😔 Поки немає збережених новин. Спробуй пізніше.", reply_markup=main_keyboard(),
        )
        await callback.answer()
        return

    cards = []
    for p in posts:
        pros_lines = "\n".join(f"• {html.escape(pro)}" for pro in p["pros"])
        cards.append(
            f"🤖 <b>{html.escape(p['name'])}</b>\n\n"
            f"📝 {html.escape(p['desc'])}\n\n"
            f"💼 <b>Переваги:</b>\n{pros_lines}\n\n"
            f"💡 <b>Промпт:</b> <i>{html.escape(p['prompt'])}</i>\n\n"
            f"<b><a href='{p['url']}'>Відкрити інструмент →</a></b>"
        )

    divider = "\n\n<b>━━━━━━━━━━━━━━</b>\n\n"
    await callback.message.answer(
        f"⚡️ <b>AI-ДАЙДЖЕСТ</b>\n<b>━━━━━━━━━━━━━━</b>\n\n" + divider.join(cards),
        parse_mode="HTML", disable_web_page_preview=True,
    )
    await callback.message.answer("Чим ще можу допомогти?", reply_markup=main_keyboard())
    await callback.answer()


# --- Tool search ---

@dp.callback_query(F.data == "search")
async def cb_search(callback: CallbackQuery, state: FSMContext):
    await state.set_state(SearchState.waiting_for_query)
    await callback.message.answer(
        "🔎 <b>Пошук ШІ-інструментів</b>\n\n"
        "Введи назву або опис того, що шукаєш:\n"
        "<i>Приклади: image generator, code assistant, Midjourney</i>",
        parse_mode="HTML", reply_markup=cancel_keyboard(),
    )
    await callback.answer()


@dp.message(SearchState.waiting_for_query, F.text == "❌ Скасувати")
async def cancel_search(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Скасовано 👌", reply_markup=ReplyKeyboardRemove())
    await message.answer("Обери розділ:", reply_markup=main_keyboard())


@dp.message(SearchState.waiting_for_query, F.text)
async def process_search(message: Message, state: FSMContext):
    if err := validate_input(message.text):
        await message.answer(err, reply_markup=ReplyKeyboardRemove())
        await state.clear()
        return

    await state.clear()
    query   = message.text.strip()
    results = await search_news(query, limit=5)

    if not results:
        await message.answer(
            f"😔 За запитом <i>«{html.escape(query)}»</i> нічого не знайдено.",
            parse_mode="HTML", reply_markup=ReplyKeyboardRemove(),
        )
        await message.answer("Обери розділ:", reply_markup=main_keyboard())
        return

    cards = []
    for r in results:
        pros_lines = "\n".join(f"• {html.escape(pro)}" for pro in r["pros"])
        cards.append(
            f"🤖 <b>{html.escape(r['name'])}</b>\n\n"
            f"📝 {html.escape(r['desc'])}\n\n"
            f"💼 <b>Переваги:</b>\n{pros_lines}\n\n"
            f"<b><a href='{r['url']}'>Відкрити інструмент →</a></b>"
        )

    divider = "\n\n<b>━━━━━━━━━━━━━━</b>\n\n"
    await message.answer(
        f"🔎 <b>РЕЗУЛЬТАТИ ПОШУКУ</b>\n"
        f"Запит: <i>«{html.escape(query)}»</i>\n"
        f"<b>━━━━━━━━━━━━━━</b>\n\n" + divider.join(cards),
        parse_mode="HTML", disable_web_page_preview=True, reply_markup=ReplyKeyboardRemove(),
    )
    await message.answer("Повертаюся до головного меню 👇", reply_markup=main_keyboard())


# --- Prompt trainer ---

@dp.callback_query(F.data == "prompt_trainer")
async def cb_prompt_trainer(callback: CallbackQuery, state: FSMContext):
    await state.set_state(PromptState.waiting_for_prompt)
    await callback.message.answer(
        "🪄 <b>ШІ-тренер промптів</b>\n\n"
        "Надішли свій промпт — я покращу його та поясню, що змінив.\n\n"
        "<i>Приклад: «Напиши текст про каву»</i>",
        parse_mode="HTML", reply_markup=cancel_keyboard(),
    )
    await callback.answer()


@dp.message(PromptState.waiting_for_prompt, F.text == "❌ Скасувати")
async def cancel_prompt(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Скасовано 👌", reply_markup=ReplyKeyboardRemove())
    await message.answer("Обери розділ:", reply_markup=main_keyboard())


@dp.message(PromptState.waiting_for_prompt, F.text)
async def process_prompt(message: Message, state: FSMContext):
    if err := validate_input(message.text):
        await message.answer(err, reply_markup=ReplyKeyboardRemove())
        await state.clear()
        return

    await state.clear()
    await message.answer("🔄 Аналізую промпт...", reply_markup=ReplyKeyboardRemove())
    await message.bot.send_chat_action(message.chat.id, action="typing")

    result = await improve_prompt(message.text)
    await message.answer(result, parse_mode="HTML")
    await message.answer("Чим ще можу допомогти?", reply_markup=main_keyboard())


# --- Model comparison ---

@dp.callback_query(F.data == "compare_ai")
async def cb_compare_ai(callback: CallbackQuery, state: FSMContext):
    await state.set_state(CompareState.waiting_for_models)
    await callback.message.answer(
        "📊 <b>Порівняння нейромереж</b>\n\n"
        "Введіть назви двох нейромереж через кому або пробіл:\n\n"
        "<i>Приклади: ChatGPT, Gemini  |  Midjourney DALL-E</i>",
        parse_mode="HTML", reply_markup=cancel_keyboard(),
    )
    await callback.answer()


@dp.message(CompareState.waiting_for_models, F.text == "❌ Скасувати")
async def cancel_compare(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Скасовано 👌", reply_markup=ReplyKeyboardRemove())
    await message.answer("Обери розділ:", reply_markup=main_keyboard())


@dp.message(CompareState.waiting_for_models, F.text)
async def process_compare(message: Message, state: FSMContext):
    if err := validate_input(message.text):
        await message.answer(err, reply_markup=ReplyKeyboardRemove())
        await state.clear()
        return

    await state.clear()
    raw   = message.text.strip()
    parts = [p.strip() for p in raw.split(",", 1)] if "," in raw else raw.split(None, 1)

    if len(parts) < 2 or not parts[1]:
        await message.answer(
            "⚠️ Не вдалося розпізнати дві назви. Спробуй ще раз.\n"
            "<i>Приклад: ChatGPT, Gemini</i>",
            parse_mode="HTML", reply_markup=ReplyKeyboardRemove(),
        )
        await message.answer("Обери розділ:", reply_markup=main_keyboard())
        return

    ai1, ai2 = parts[0], parts[1]
    await message.answer(
        f"🔄 Порівнюю <b>{html.escape(ai1)}</b> та <b>{html.escape(ai2)}</b>...",
        parse_mode="HTML", reply_markup=ReplyKeyboardRemove(),
    )
    await message.bot.send_chat_action(message.chat.id, action="typing")

    result = await compare_ai(ai1, ai2)
    await message.answer(result, parse_mode="HTML")
    await message.answer("Чим ще можу допомогти?", reply_markup=main_keyboard())


# --- Voice handler ---

@dp.message(F.content_type == ContentType.VOICE)
async def handle_voice(message: Message):
    MAX_VOICE_SIZE = 25 * 1024 * 1024
    if message.voice.file_size > MAX_VOICE_SIZE:
        await message.answer("⚠️ Голосове повідомлення надто велике (макс. 25 МБ).")
        return

    await message.answer("🎙 Чую тебе! Розпізнаю мовлення...")
    await message.bot.send_chat_action(message.chat.id, action="typing")

    file     = await message.bot.get_file(message.voice.file_id)
    tmp_path = os.path.join(TEMP_DIR, f"voice_{message.from_user.id}_{message.message_id}.ogg")  # cross-platform path
    await message.bot.download_file(file.file_path, destination=tmp_path)

    result = await transcribe_and_answer(tmp_path, message.from_user.id)
    await message.answer(result, parse_mode="HTML")


# --- Chat handler ---

@dp.message(F.text, ~F.text.startswith("/"))
async def handle_chat(message: Message, state: FSMContext):
    if await state.get_state() is not None:  # yield to FSM handlers
        return

    if err := validate_input(message.text):
        await message.answer(err)
        return

    await message.bot.send_chat_action(message.chat.id, action="typing")
    reply = await ask_groq(message.from_user.id, message.text)
    await message.answer(reply, parse_mode="HTML")


# --- Admin commands ---

@dp.message(Command("force_update"))
async def force_update(message: Message):
    if message.from_user.id != ADMIN_ID:
        return await message.answer("⛔ Немає доступу.")
    status_msg = await message.answer("🔄 Починаю пошук та обробку свіжих нейромереж...")
    saved = await run_pipeline()
    await status_msg.edit_text(
        f"✅ Базу оновлено! Нових записів: <b>{saved}</b>.\n"
        "Свіжі підбірки доступні за кнопкою в меню.",
        parse_mode="HTML",
    )


@dp.message(Command("crash"))
async def cmd_crash(message: Message):
    if message.from_user.id != ADMIN_ID:
        return await message.answer("⛔ Немає доступу.")
    await message.answer("💥 Запускаю тестовий краш...")
    _ = 1 / 0  # triggers global_error_handler


# --- Startup ---

async def scheduled_pipeline():
    logging.info("Запуск планової перевірки RSS...")
    await run_pipeline()


async def main():
    await init_db()

    dp.message.middleware(ThrottlingMiddleware())
    logging.info("✅ ThrottlingMiddleware зареєстровано (ліміт: 2 сек)")

    dp.errors.register(global_error_handler)
    logging.info("✅ Error handler зареєстровано")

    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_pipeline, "interval", hours=6)
    scheduler.start()

    logging.info("🚀 Бот Mindflow AI запущено")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())