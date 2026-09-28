"""Telegram-бот: принимает сообщения клиентов и отвечает через агента.

Запуск: python telegram.py
Настройки (токен и тексты) — в telegram.cfg.
"""
import asyncio
import configparser
import sys
import traceback
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.types import Message

from agent import Agent
from kb import KnowledgeBase
from schema import load_studio

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "telegram.cfg"
TELEGRAM_LIMIT = 4096   # максимальная длина одного сообщения в Telegram
TYPING_INTERVAL = 4     # «печатает…» гаснет через 5 с, поэтому обновляем чаще

# ---------- настройки ----------

config = configparser.ConfigParser(interpolation=None)  # без интерполяции: «%» в текстах не ломает чтение
if not config.read(CONFIG_FILE, encoding="utf-8"):
    sys.exit(f"Не найден файл настроек {CONFIG_FILE}")

TOKEN = config["telegram"]["token"].strip()
if not TOKEN:
    sys.exit("В telegram.cfg не задан token. Получите его у @BotFather (команда /newbot).")
ALLOWED_USERS = {int(x) for x in config["telegram"].get("allowed_users", "").replace(",", " ").split()}

bot_cfg = config["bot"]
STUDIO_FILE = BASE_DIR / bot_cfg["studio_file"]
START_MESSAGE = bot_cfg["start_message"]
ERROR_MESSAGE = bot_cfg["error_message"]
NOT_TEXT_MESSAGE = bot_cfg["not_text_message"]
NOT_ALLOWED_MESSAGE = bot_cfg["not_allowed_message"]

# ---------- состояние ----------

dp = Dispatcher()
kb: KnowledgeBase | None = None
chat_fn = None
agents: dict[int, Agent] = {}   # у каждого чата свой агент со своей историей диалога
model_lock = asyncio.Lock()     # модель на одной видеокарте: запросы обрабатываются по очереди


def get_agent(chat_id: int) -> Agent:
    if chat_id not in agents:
        agents[chat_id] = Agent(kb, chat_fn, name=f"tg{chat_id}")
    return agents[chat_id]


def is_allowed(message: Message) -> bool:
    return not ALLOWED_USERS or message.from_user.id in ALLOWED_USERS


def split_message(text: str) -> list[str]:
    """Режет длинный ответ на части не длиннее лимита Telegram, по абзацам и строкам."""
    parts = []
    while len(text) > TELEGRAM_LIMIT:
        cut = text.rfind("\n", 0, TELEGRAM_LIMIT)
        if cut <= 0:
            cut = TELEGRAM_LIMIT
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    parts.append(text)
    return parts


async def keep_typing(bot: Bot, chat_id: int) -> None:
    """Показывает «печатает…», пока модель готовит ответ."""
    while True:
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
        await asyncio.sleep(TYPING_INTERVAL)


# ---------- обработчики ----------

@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    if not is_allowed(message):
        await message.answer(NOT_ALLOWED_MESSAGE.format(user_id=message.from_user.id))
        return
    agents.pop(message.chat.id, None)  # /start начинает диалог заново
    await message.answer(START_MESSAGE)


@dp.message(F.text)
async def on_text(message: Message) -> None:
    if not is_allowed(message):
        await message.answer(NOT_ALLOWED_MESSAGE.format(user_id=message.from_user.id))
        return

    typing = asyncio.create_task(keep_typing(message.bot, message.chat.id))
    try:
        async with model_lock:
            # генерация блокирующая — выполняем в отдельном потоке, чтобы бот не «замирал»
            answer = await asyncio.to_thread(get_agent(message.chat.id).ask, message.text)
    except Exception:
        traceback.print_exc()
        answer = ERROR_MESSAGE
    finally:
        typing.cancel()

    for part in split_message(answer or ERROR_MESSAGE):
        await message.answer(part)


@dp.message()
async def on_other(message: Message) -> None:
    if is_allowed(message):
        await message.answer(NOT_TEXT_MESSAGE)


# ---------- запуск ----------

async def main() -> None:
    global kb, chat_fn
    kb = KnowledgeBase(load_studio(STUDIO_FILE))
    from cloud_inference import chat  # подключение модели — после проверки настроек (локальная: from inference import chat)
    chat_fn = chat

    bot = Bot(TOKEN)
    me = await bot.get_me()
    print(f"Бот @{me.username} запущен, студия «{kb.studio.name}». Остановка: Ctrl+C")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
