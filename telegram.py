"""Telegram-бот: принимает сообщения клиентов и отвечает через агента.

Запуск: python telegram.py
Настройки (токен и тексты) — в telegram.cfg.
"""
import asyncio
import configparser
import logging
import sys
import traceback
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, User

import bookings
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
_admin_chat = bot_cfg.get("admin_chat_id", "").strip()
ADMIN_CHAT_ID = int(_admin_chat) if _admin_chat else None  # куда слать заявки; пусто — никуда

# ---------- состояние ----------

dp = Dispatcher()
kb: KnowledgeBase | None = None
chat_fn = None
agents: dict[int, Agent] = {}   # у каждого чата свой агент со своей историей диалога
model_lock = asyncio.Lock()     # сообщения обрабатываются по очереди


def get_agent(chat_id: int) -> Agent:
    if chat_id not in agents:
        agents[chat_id] = Agent(kb, chat_fn, name=f"tg{chat_id}")
    return agents[chat_id]


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USERS or user_id in ALLOWED_USERS


def confirm_keyboard(booking_id: str) -> InlineKeyboardMarkup:
    """Кнопки под подтверждением заявки. id заявки в кнопке — чтобы старая кнопка не подтвердила новую заявку."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Да, всё верно", callback_data=f"booking:yes:{booking_id}"),
        InlineKeyboardButton(text="Нет, исправить", callback_data=f"booking:no:{booking_id}"),
    ]])


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


async def notify_admin(bot: Bot, agent: Agent, user: User) -> None:
    """Переслать администратору заявки, которые агент только что сохранил."""
    while agent.new_bookings:
        booking = agent.new_bookings.pop(0)
        if ADMIN_CHAT_ID is None:
            continue
        contact = f"Telegram: @{user.username}" if user.username else f"Telegram: {user.full_name} (id {user.id})"
        try:
            await bot.send_message(ADMIN_CHAT_ID, bookings.admin_text(booking, contact))
        except Exception as e:  # клиенту это не мешает: заявка уже в bookings.json
            logging.getLogger("bot").error("[%s] НЕ УДАЛОСЬ УВЕДОМИТЬ АДМИНИСТРАТОРА о заявке %s: %s",
                                           agent.name, booking.id, e)
            traceback.print_exc()


async def keep_typing(bot: Bot, chat_id: int) -> None:
    """Показывает «печатает…», пока модель готовит ответ."""
    while True:
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
        await asyncio.sleep(TYPING_INTERVAL)


# ---------- обработчики ----------

@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer(NOT_ALLOWED_MESSAGE.format(user_id=message.from_user.id))
        return
    agents.pop(message.chat.id, None)  # /start начинает диалог заново
    await message.answer(START_MESSAGE)


@dp.message(Command("id"))
async def on_id(message: Message) -> None:
    """Номер чата — чтобы вписать его в admin_chat_id (работает и в группах)."""
    await message.answer(f"id этого чата: {message.chat.id}")


@dp.message(F.text)
async def on_text(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer(NOT_ALLOWED_MESSAGE.format(user_id=message.from_user.id))
        return

    agent = get_agent(message.chat.id)
    typing = asyncio.create_task(keep_typing(message.bot, message.chat.id))
    try:
        async with model_lock:
            # генерация блокирующая — выполняем в отдельном потоке, чтобы бот не «замирал»
            answer = await asyncio.to_thread(agent.ask, message.text)
    except Exception:
        traceback.print_exc()
        answer = ERROR_MESSAGE
    finally:
        typing.cancel()

    parts = split_message(answer or ERROR_MESSAGE)
    for part in parts[:-1]:
        await message.answer(part)
    # если бот показал заявку на подтверждение — добавляем кнопки «да / нет»
    markup = confirm_keyboard(agent.pending.id) if agent.awaiting_confirmation else None
    await message.answer(parts[-1], reply_markup=markup)
    await notify_admin(message.bot, agent, message.from_user)  # если клиент подтвердил заявку текстом «да»


@dp.callback_query(F.data.startswith("booking:"))
async def on_booking_button(callback: CallbackQuery) -> None:
    """Нажатие «Да, всё верно» / «Нет, исправить» под заявкой. Без LLM."""
    if not is_allowed(callback.from_user.id):
        await callback.answer()
        return
    _, choice, booking_id = callback.data.split(":", 2)
    await callback.message.edit_reply_markup(reply_markup=None)  # убираем кнопки, чтобы не нажали дважды
    agent = agents.get(callback.message.chat.id)
    if agent is None or agent.pending is None or agent.pending.id != booking_id:
        await callback.answer("Эта заявка уже обработана")
        return
    async with model_lock:
        answer = agent.confirm(choice == "yes")
    await callback.answer()
    await callback.message.answer(answer)
    await notify_admin(callback.bot, agent, callback.from_user)


@dp.message()
async def on_other(message: Message) -> None:
    if is_allowed(message.from_user.id):
        await message.answer(NOT_TEXT_MESSAGE)


# ---------- запуск ----------

async def main() -> None:
    global kb, chat_fn
    kb = KnowledgeBase(load_studio(STUDIO_FILE))
    from cloud_inference import chat  # подключение модели — после проверки настроек
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
