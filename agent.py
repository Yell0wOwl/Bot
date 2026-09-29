"""Ядро бота: системный промпт, история диалога и цикл вызова инструментов.

Не зависит от канала (консоль, Telegram) и от модели: модель подключается
через функцию chat(), которую предоставляет cloud_inference.py.

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
Как перевести историю в формат конкретной модели и разобрать её ответ — забота cloud_inference.py.
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

import bookings
from kb import KnowledgeBase


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
NOT_WANTED = "Клиент отказался от записи — заявку не создавай, просто ответь ему."
ALREADY_SHOWN = ("Эта заявка уже показана клиенту и ждёт ответа «да» или «нет». Не повторяй её и не пиши, что она "
                 "оформлена или принята: клиент её ещё не подтвердил. Ответь на вопрос клиента, если он есть, — "
                 "заявку система покажет сама после твоего ответа.")
# модель не должна сама объявлять заявку оформленной: это делает только код после «да»
BOOKING_CLAIM_RE = re.compile(r"заявк\w*\s+(?:уже\s+)?(?:оформлен|принят|подтвержд|создан|сохранен|сохранён)")

SYSTEM_PROMPT = """Ты — администратор компании «{name}» и отвечаешь клиентам в чате.

ПРАВИЛА
1. Факты об услугах, ценах, адресе и графике бери только из разделов «О компании», «Сегодня», каталога и ответов инструментов. Чего там нет — того ты не знаешь. Не объясняй от себя, что входит в работы и как они проходят («после осмотра», «мастер проверит»), не придумывай описаний компании и вариантов, которых нет в данных («автотехцентр», «партнёры»).
2. Цены — только через get_prices. Называй цену дословно из поля price: ничего не считай, не округляй и не пересчитывай — надбавки по словам клиента уже учтены. Надбавки из поля surcharges называй как есть («+10%»). Если есть поле assumed — скажи, для какого варианта названа цена.
3. В get_prices передавай только те параметры, которые клиент назвал или которые однозначно следуют из его слов по разделу «Параметры цены». Не угадывай: если модели машины нет в примерах — параметр не передавай.
4. Если в ответе есть depends_on — цена зависит от того, чего клиент не сказал: назови диапазон из price и задай вопрос из depends_on.
5. Если price — «УТОЧНИТЬ У АДМИНИСТРАТОРА» или есть поле admin_for — эту цену не называй. Если клиент спросил о цене — скажи, что её подскажет администратор, и дай телефон. Если о цене не спрашивали — не упоминай ни цену, ни администратора, даже если вызывал get_prices.
6. Если клиент спрашивает о том, чего нет в данных — скидки, акции, свободное время, услуги не из каталога, материалы, сроки, гарантия на конкретную работу, — не отказывай и не утверждай ничего от себя: скажи, что это подскажет администратор, и дай телефон. Про услугу не из каталога не говори «у нас этого нет» — скажи, что в каталоге её нет, а возможность уточнит администратор. Если в каталоге есть похожая услуга — упомяни её.
7. Срок работы называй только если он есть в поле duration — иначе его подскажет администратор или мастер.
8. Время работы («открыты ли завтра», «работаете сейчас», «в субботу») — отвечай по разделу «Сегодня». Даты и дни недели сам не вычисляй. Про праздники в данных ничего нет — к администратору.
9. Запись. Если клиент хочет записаться, узнай имя, номер телефона, желаемые дату и время и услугу (что уже ясно из диалога — не переспрашивай). Не придумывай данные за клиента. Когда всё известно — вызови create_booking (service_id — id из каталога, если услуга оттуда; время — словами клиента, дату сам не вычисляй). Подтверждение клиенту покажет и заявку сохранит система: сам заявку не подтверждай и конкретное время не обещай — время подтвердит администратор. Если create_booking вернул ошибку — уточни у клиента то, что в ней указано.
10. Если вопрос не про компанию и её услуги — вежливо верни разговор к услугам.

СТИЛЬ
Пиши по-русски, на «вы», коротко: 1–4 предложения. Без таблиц и без markdown-заголовков. Если вариантов несколько — короткий список. Отвечай только на заданный вопрос: про цену, хранение, магазин и другие услуги говори, только если клиент о них спросил; телефон администратора давай, когда отправляешь к нему. Предлагай записаться, только если клиент интересуется конкретной услугой и ещё не записывается, — не в каждом ответе.

О КОМПАНИИ
{info}

ПАРАМЕТРЫ ЦЕНЫ
{parameters}

СЕГОДНЯ
<<CALENDAR>>

КАТАЛОГ (id и названия; цен здесь нет — их даёт get_prices)
{outline}"""


class Agent:
    def __init__(self, kb: KnowledgeBase, llm: ChatFn, name: str | None = None):
        """llm — функция chat(messages, tools) -> Reply из cloud_inference.py.
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
            info=kb.studio_info(),
            parameters=kb.parameters_info(),
            outline=kb.outline(),
        )
        self.system = self.system_template.replace(CALENDAR_MARK, kb.calendar(kb.now()))
        self.tools = kb.tools() + [bookings.BOOKING_TOOL]  # параметры get_prices — из параметров компании
        self.history: list[dict] = []
        self.pending: bookings.Booking | None = None  # заявка, ждущая подтверждения клиента
        self.new_bookings: list[bookings.Booking] = []  # сохранённые, но ещё не переданные администратору

    @property
    def awaiting_confirmation(self) -> bool:
        return self.pending is not None

    def reset(self) -> None:
        self.history = []
        self.pending = None

    def _run_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "get_prices":
                return self.kb.get_prices(str(args["id"]), args.get("params"))
            if name == "search_services":
                return self.kb.search_services(args["query"])
            known = ", ".join(t["function"]["name"] for t in self.tools)
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

    def _client_text(self) -> str:
        return "\n".join(m["content"] for m in self.history if m["role"] == "user")

    def cancel(self, text: str) -> str:
        """Клиент передумал записываться. Без LLM."""
        booking, self.pending = self.pending, None
        self._log(f"ЗАЯВКА ОТМЕНЕНА КЛИЕНТОМ {booking.id}")
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": bookings.CANCEL_TEXT}]
        return bookings.CANCEL_TEXT

    def confirm(self, ok: bool, text: str | None = None) -> str:
        """Ответ клиента на подтверждение заявки (кнопка или «да»/«нет»). Без LLM."""
        booking, self.pending = self.pending, None
        if booking is None:
            return "Эта заявка уже обработана."
        self.history.append({"role": "user", "content": text or ("Да, всё верно" if ok else "Нет, исправить")})
        if ok:
            bookings.save(booking)
            self.new_bookings.append(booking)  # канал (Telegram) заберёт и уведомит администратора
            answer = bookings.saved_text(booking)
            self._log(f"ЗАЯВКА СОХРАНЕНА {booking.id}: {booking.name}, {booking.phone}, {booking.service}, {booking.desired_time}")
        else:
            answer = bookings.CORRECTION_TEXT
            self._log(f"ЗАЯВКА ОТКЛОНЕНА КЛИЕНТОМ {booking.id}")
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _ask(self, text: str) -> str:
        if self.pending is not None:  # ждём подтверждения заявки — сначала разбираем ответ кодом
            if bookings.is_cancel(text):
                return self.cancel(text)
            decision = bookings.parse_confirmation(text)
            if decision is not None:
                return self.confirm(decision, text)
            # вопрос или исправление: отвечает модель, заявка ждёт подтверждения, пока модель не создаст новую
            self._log(f"ЗАЯВКА {self.pending.id}: ответ не «да/нет» — передаём модели")
        self.system = self.system_template.replace(CALENDAR_MARK, self.kb.calendar(self.kb.now()))
        self.history.append({"role": "user", "content": text})
        note: list[dict] = []  # замечание проверки ответа; в историю диалога не попадает
        for _ in range(MAX_STEPS):
            messages = [{"role": "system", "content": self.system}] + self._trimmed_history() + note
            reply = self.llm(messages, self.tools)
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
                if self.pending is not None:  # заявка всё ещё ждёт ответа — напоминаем её
                    if BOOKING_CLAIM_RE.search(answer.lower()):
                        self._log(f"ТЕКСТ МОДЕЛИ УБРАН (заявка ещё не подтверждена): {answer}")
                        answer = ""
                    answer = f"{answer}\n\n{bookings.confirmation_text(self.pending)}".strip()
                self.history.append({"role": "assistant", "content": answer})
                return answer
            self.history.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                               for c in reply.tool_calls],
            })
            booking = None
            for c in reply.tool_calls:
                if c.name == "create_booking" and bookings.is_cancel(text):  # «запись не нужна»
                    booking, result = None, {"status": NOT_WANTED}
                elif c.name == "create_booking":
                    booking, problem = bookings.validate(c.arguments, self.kb.studio, self.name, self._client_text())
                    if booking and self.pending and bookings.same(booking, self.pending):
                        booking, result = None, {"status": ALREADY_SHOWN}  # не показываем ту же заявку повторно
                    else:
                        if problem:
                            self.pending = None  # клиент исправляет заявку, но данные пока неполные
                        result = problem or {"status": "Заявка показана клиенту, ждём подтверждения. Не повторяй её."}
                else:
                    result = self._run_tool(c.name, c.arguments)
                result_json = json.dumps(result, ensure_ascii=False)
                self._log(f"ИНСТРУМЕНТ {c.name}({json.dumps(c.arguments, ensure_ascii=False)}) → {result_json}")
                self.history.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                                     "content": result_json})
            if booking is not None:  # данные в порядке — подтверждение пишет код, а не модель
                self.pending = booking
                answer = bookings.confirmation_text(booking)
                self.history.append({"role": "assistant", "content": answer})
                return answer
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
