"""Ядро бота: системный промпт, история диалога и цикл вызова инструментов.

Не зависит от канала (консоль, Telegram) и от модели: модель подключается
через функцию chat(), которую предоставляет inference.py.

КОНТРАКТ С МОДЕЛЬЮ
    chat(messages: list[dict], tools: list[dict]) -> Reply

messages — история диалога, сообщения четырёх видов:
    {"role": "system",    "content": str}
    {"role": "user",      "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id": str, "name": str, "arguments": dict}]}
                          ("tool_calls" есть только если модель вызывала инструменты)
    {"role": "tool",      "tool_call_id": str, "name": str, "content": str}   # результат инструмента, JSON-строка
tools — описания инструментов в формате {"type": "function", "function": {"name", "description", "parameters"}}

chat() должна вернуть Reply: текст ответа и/или список вызовов инструментов.
Как перевести историю в формат конкретной модели и разобрать её ответ — забота inference.py.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from kb import TOOLS, KnowledgeBase


@dataclass
class ToolCall:
    id: str           # любой уникальный в пределах диалога идентификатор
    name: str         # имя инструмента
    arguments: dict   # аргументы, уже разобранные из JSON


@dataclass
class Reply:
    content: str = ""                                          # текст ответа
    tool_calls: list[ToolCall] = field(default_factory=list)   # пусто — модель ответила текстом


ChatFn = Callable[[list[dict], list[dict]], Reply]

# ---------- лог: вопросы, вызовы инструментов, ответы, время ----------

LOG_FILE = Path(__file__).parent / "log.txt"
log = logging.getLogger("bot")
if not log.handlers:
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _oneline(text: str) -> str:
    """Многострочный текст в логе: продолжение строк с отступом, чтобы записи не сливались."""
    return text.replace("\n", "\n    ")

MAX_STEPS = 6       # сколько раз подряд модель может вызвать инструменты за один ответ
MAX_HISTORY = 30    # сколько последних сообщений диалога держим в контексте
CALENDAR_MARK = "<<CALENDAR>>"
MIN_CHECKED_NUMBER = 100  # числа меньше (классы кузова, проценты, часы) не проверяем

# «15 000», «15000», «1 500 000» -> одно число (пробелы только между тройками цифр,
# чтобы «3 класса 28 000» не склеилось в 328000); неразрывные пробелы тоже считаются
NUMBER_RE = re.compile(r"\d{1,3}(?:[   ]\d{3})+|\d+")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)

RETRY_NOTE = ("Проверка ответа: числа {numbers} отсутствуют в данных студии и в результатах инструментов. "
              "Не выдумывай цены. Вызови get_prices для нужной позиции каталога и назови цену дословно из результата.")
FALLBACK_PRICE = "Не могу точно назвать стоимость. Её подскажет администратор{phone}."

SYSTEM_PROMPT = """Ты — администратор детейлинг-студии «{name}» и отвечаешь клиентам в чате.

ПРАВИЛА
1. Факты об услугах, ценах, адресе и графике бери только из раздела «О студии», каталога и ответов инструментов. Чего там нет — того ты не знаешь.
2. Перед тем как назвать любую цену, вызови get_prices. Называй цену дословно из поля price или range_all_classes. Не считай, не округляй и не пересчитывай надбавки сам — называй их как есть, например «+10% за матовую плёнку».
3. Если в price написано «НЕ НАЗЫВАТЬ» — не называй цену, скажи, что стоимость уточнит администратор.
4. Класс кузова. Студия делит машины на классы: {body_classes}. Не определяй класс по марке машины сам — ты не знаешь, как студия их распределяет. Если класс неизвестен, называй диапазон из range_all_classes и говори, что точную цену для конкретной машины подтвердит администратор.
5. Если duration не указан — не называй срок, скажи, что он зависит от состояния машины и его подскажет мастер.
5а. Вопросы про время работы («открыты ли завтра», «работаете сейчас», «в субботу») — отвечай по разделу «Сегодня». Дни недели и даты сам не вычисляй. Про праздничные дни в данных ничего нет — пусть уточнят по телефону.
6. Не обещай скидок, акций и свободного времени для записи.
7. Если услуги нет в каталоге — скажи честно и предложи похожие из каталога или позвонить в студию.
8. Если клиент готов записаться — попроси позвонить по телефону студии (записывать сам ты пока не умеешь).
9. Если вопрос не про студию и не про уход за машиной — вежливо верни разговор к услугам студии.

СТИЛЬ
Пиши по-русски, на «вы», коротко: 1–4 предложения. Без таблиц и без markdown-заголовков. Если вариантов несколько — короткий список. Когда уместно, заканчивай вопросом, который двигает к записи.

О СТУДИИ
{info}

СЕГОДНЯ
<<CALENDAR>>

КАТАЛОГ (id и названия; цен здесь нет — их даёт get_prices)
{outline}"""


class Agent:
    def __init__(self, kb: KnowledgeBase, llm: ChatFn, name: str | None = None):
        """llm — функция chat(messages, tools) -> Reply из inference.py / cloud_inference.py.
        name — метка диалога в логе (например, id чата в Telegram); по умолчанию случайная."""
        self.kb = kb
        self.llm = llm
        self.name = name or uuid.uuid4().hex[:6]
        self.last_duration: float | None = None  # время последнего ответа, секунды
        st = kb.studio
        # календарь («сегодня», «завтра», «сейчас открыто») подставляется при каждом вопросе:
        # бот в Telegram работает сутками, и дата должна быть текущей
        self.system_template = SYSTEM_PROMPT.format(
            name=st.name,
            body_classes=", ".join(f"{b.name} (код {b.code})" for b in st.body_classes) or "не указаны",
            info=kb.studio_info(),
            outline=kb.outline(),
        )
        self.system = self.system_template.replace(CALENDAR_MARK, kb.calendar(kb.now()))
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history = []

    def _run_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "get_prices":
                return self.kb.get_prices(args["service_class_id"], args.get("body_class"))
            if name == "search_services":
                return self.kb.search_services(args["query"])
            known = ", ".join(t["function"]["name"] for t in TOOLS)
            return {"error": f"Нет инструмента «{name}» или вызов не разобран. Доступные: {known}. "
                             f"Вызов должен быть JSON с полями name и arguments."}
        except Exception as e:  # ошибку отдаём модели, а не роняем бота
            return {"error": f"{type(e).__name__}: {e}"}

    def ask(self, text: str) -> str:
        """Ответ на сообщение клиента. Всё происходящее пишется в log.txt."""
        start = time.monotonic()
        self._log(f"ВОПРОС: {text}")
        try:
            answer = self._ask(text)
        except Exception as e:
            self._log(f"ОШИБКА через {time.monotonic() - start:.1f} с: {type(e).__name__}: {e}")
            raise
        self.last_duration = time.monotonic() - start
        self._log(f"ОТВЕТ ({self.last_duration:.1f} с): {answer}")
        return answer

    def _log(self, message: str) -> None:
        log.info("[%s] %s", self.name, _oneline(message))

    def _ask(self, text: str) -> str:
        self.system = self.system_template.replace(CALENDAR_MARK, self.kb.calendar(self.kb.now()))
        self.history.append({"role": "user", "content": text})
        note: list[dict] = []  # замечание проверки ответа; в историю диалога не попадает
        for _ in range(MAX_STEPS):
            messages = [{"role": "system", "content": self.system}] + self._trimmed_history() + note
            reply = self.llm(messages, TOOLS)
            if not reply.tool_calls:
                answer = reply.content.strip()
                bad = self._unverified_numbers(answer)
                if bad and not note:  # первая попытка — даём модели исправиться
                    self._log(f"ОТВЕТ ОТКЛОНЁН (числа не из данных: {', '.join(bad)}): {answer}")
                    note = [{"role": "system", "content": RETRY_NOTE.format(numbers=", ".join(bad))}]
                    continue
                if bad:               # не исправилась — не отдаём клиенту выдуманную цену
                    self._log(f"ОТВЕТ ОТКЛОНЁН ПОВТОРНО (числа не из данных: {', '.join(bad)}): {answer}")
                    answer = self._fallback_answer()
                self.history.append({"role": "assistant", "content": answer})
                return answer
            self.history.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                               for c in reply.tool_calls],
            })
            for c in reply.tool_calls:
                result = self._run_tool(c.name, c.arguments)
                result_json = json.dumps(result, ensure_ascii=False)
                self._log(f"ИНСТРУМЕНТ {c.name}({json.dumps(c.arguments, ensure_ascii=False)}) → {result_json}")
                self.history.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                                     "content": result_json})
        self._log(f"ЛИМИТ ШАГОВ: модель {MAX_STEPS} раз вызывала инструменты, не дав ответа")
        fallback = "Извините, не получилось найти ответ. Уточните, пожалуйста, вопрос или позвоните нам."
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    def _unverified_numbers(self, answer: str) -> list[str]:
        """Числа из ответа, которых нет ни в данных студии, ни в результатах инструментов,
        ни в словах клиента. Если такие есть — модель, скорее всего, выдумала цену."""
        sources = [self.system] + [m["content"] for m in self.history if m["role"] in ("tool", "user")]
        known = {_digits(n) for s in sources for n in NUMBER_RE.findall(s)}
        long_known = [k for k in known if len(k) >= 7]  # телефоны: в ответе их часто пишут по частям
        bad = []
        for raw in NUMBER_RE.findall(answer):
            d = _digits(raw)
            if int(d) < MIN_CHECKED_NUMBER or d in known or any(d in k for k in long_known):
                continue
            bad.append(raw)
        return bad

    def _fallback_answer(self) -> str:
        phones = self.kb.studio.contacts.phone
        return FALLBACK_PRICE.format(phone=f" по телефону {phones[0]}" if phones else "")

    def _trimmed_history(self) -> list[dict]:
        """Последние MAX_HISTORY сообщений, но не с середины вызова инструмента:
        история должна начинаться с сообщения пользователя."""
        h = self.history[-MAX_HISTORY:]
        while h and h[0]["role"] != "user":
            h = h[1:]
        return h
