"""Заявки на запись: проверка данных, текст подтверждения, разбор ответа клиента, сохранение.

Здесь нет LLM: модель только собирает данные и вызывает инструмент create_booking,
а подтверждение, решение «да/нет» и запись в bookings.json делает этот код.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from kb import DAY_KEYS, MONTHS_RU, WEEKDAYS_RU
from schema import SEP, Studio

BOOKINGS_FILE = Path(__file__).parent / "bookings.json"
_file_lock = threading.Lock()

CONFIRM_TEMPLATE = ("Проверьте, пожалуйста, заявку:\n"
                    "Имя: {name}\n"
                    "Телефон: {phone}\n"
                    "Услуга: {service}\n"
                    "Желаемое время: {desired_time}\n"
                    "{comment}"
                    "Всё верно? Ответьте «да» или «нет».")
SAVED_TEXT = ("Спасибо, {name}! Заявка принята. Администратор свяжется с вами по номеру {phone}, "
              "чтобы подтвердить время.")
CORRECTION_TEXT = "Хорошо, что нужно исправить?"
CANCEL_TEXT = "Хорошо, заявку отменил. Если решите записаться — просто напишите."
NOT_IN_CATALOG = " (нет в каталоге — возможность уточнит администратор)"
ADMIN_TEMPLATE = ("Новая заявка {id}\n"
                  "Имя: {name}\n"
                  "Телефон: {phone}\n"
                  "Услуга: {service}\n"
                  "Желаемое время: {desired_time}\n"
                  "{comment}"
                  "{contact}"
                  "Создана: {created_at}")

YES_WORDS = {"да", "верно", "правильно", "подтверждаю", "подтвердить", "ок", "ok", "ага", "угу", "yes", "+"}
NO_WORDS = {"нет", "неверно", "неправильно", "исправить", "изменить", "ошибка", "no", "-"}
NEGATIONS = {"не", "нет"}
CANCEL_WORDS = {"передумал", "передумала", "отмена", "отменить", "отмените", "отменяю", "отбой"}
CANCEL_PHRASES = ("не нужно", "не нужна", "не нужен", "не надо", "не буду записываться")

WEEKDAY_RE = re.compile(r"\b(понедельник|вторник|сред[аеуы]|четверг|пятниц[аеуы]|суббот[аеуы]|воскресень[еяю])")
WEEKDAY_STEMS = ["понед", "втор", "сред", "четв", "пятн", "субб", "воск"]
DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\b")
# слова дня в desired_time («завтра», «в среду», «2 октября») должны быть и в словах клиента
DAY_WORD_RE = re.compile(r"\b(послезавтра|завтра|сегодня|" + "|".join(WEEKDAY_STEMS + [m[:3] for m in MONTHS_RU[1:]]) + ")")
MONTH_DATE_RE = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS_RU[1:]) + r")\b")

BOOKING_TOOL = {
    "type": "function",
    "function": {
        "name": "create_booking",
        "description": "Создать заявку на запись, когда клиент хочет записаться и известны имя, телефон, "
                       "желаемые день и время и услуга. Система сама покажет клиенту данные и спросит подтверждение. "
                       "Не вызывай, если клиент отказался от записи.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Имя клиента, как он представился"},
                "phone": {"type": "string", "description": "Телефон клиента, как он его написал"},
                "desired_time": {"type": "string", "description": "Желаемые дата и время дословно словами клиента, например «в четверг в 14:00». "
                                                                  "Дату сам не вычисляй (система сделает это) и не добавляй времени, которого клиент не называл."},
                "service_id": {"type": "string", "description": "id услуги или категории из каталога, если услуга оттуда; несколько услуг — id через запятую"},
                "service": {"type": "string", "description": "Услуга словами клиента, если её нет в каталоге"},
                "comment": {"type": "string", "description": "Необязательно: автомобиль, пожелания. Услугу сюда не повторяй."},
            },
            "required": ["name", "phone", "desired_time"],
        },
    },
}


@dataclass
class Booking:
    id: str
    created_at: str
    source: str          # откуда заявка: метка диалога (tg<chat_id>, test…)
    name: str
    phone: str
    service: str
    service_id: str | None
    desired_time: str    # словами клиента: «в четверг в 14:00»
    date: str | None     # дата, вычисленная кодом из desired_time (ГГГГ-ММ-ДД); None — не удалось
    comment: str | None


def normalize_phone(raw: str) -> str | None:
    """Телефон -> +7XXXXXXXXXX (или +<код страны>…). None — не похоже на номер."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 11 and digits[0] in "78":
        return "+7" + digits[1:]
    if len(digits) == 10 and digits[0] == "9":
        return "+7" + digits
    if (raw or "").strip().startswith("+") and 10 <= len(digits) <= 15:
        return "+" + digits
    return None


def display_phone(phone: str) -> str:
    d = phone.lstrip("+")
    if len(d) == 11 and d.startswith("7"):
        return f"+7 ({d[1:4]}) {d[4:7]}-{d[7:9]}-{d[9:]}"
    return phone


def _numbers(text: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", text)}


def _lower(text: str) -> str:
    return text.lower().replace("ё", "е")


def resolve_date(text: str, today: date) -> date | None:
    """«завтра», «в среду», «во вторник на следующей неделе», «2.10», «2 октября» -> дата. Без LLM."""
    t = _lower(text)
    if m := DATE_RE.search(t) or MONTH_DATE_RE.search(t):
        day = int(m[1])
        month = int(m[2]) if m[2].isdigit() else MONTHS_RU.index(m[2])
        try:
            d = date(today.year, month, day)
        except ValueError:
            return None
        return d if d >= today else d.replace(year=today.year + 1)
    for word, shift in (("послезавтра", 2), ("завтра", 1), ("сегодня", 0)):
        if word in t:
            return today + timedelta(days=shift)
    if m := WEEKDAY_RE.search(t):
        wd = next(i for i, stem in enumerate(WEEKDAY_STEMS) if m[1].startswith(stem))
        if "следующ" in t:  # «на следующей неделе» — день следующей календарной недели
            return today + timedelta(days=7 - today.weekday() + wd)
        return today + timedelta(days=(wd - today.weekday()) % 7 or 7)
    return None


def date_text(d: date) -> str:
    return f"{WEEKDAYS_RU[d.weekday()]}, {d.day} {MONTHS_RU[d.month]}"


def validate(args: dict, studio: Studio, source: str, client_text: str = "") -> tuple[Booking | None, dict | None]:
    """Данные от модели -> Booking или описание, чего не хватает (его получит модель).
    client_text — все сообщения клиента: имя, телефон и цифры времени должны быть из них,
    чтобы модель не могла подставить данные, которых клиент не называл."""
    missing, invalid = [], []
    said = _lower(client_text)
    name = str(args.get("name") or "").strip()
    if not name:
        missing.append("имя")
    elif len(name) > 60 or not re.search(r"[A-Za-zА-Яа-яЁё]", name):
        invalid.append("имя")
    elif _lower(name)[:3] not in said:
        invalid.append(f"имя «{name}» — клиент его не называл, спроси имя")

    raw_phone = str(args.get("phone") or "").strip()
    phone = normalize_phone(raw_phone)
    if not raw_phone:
        missing.append("телефон")
    elif phone is None:
        invalid.append(f"телефон «{raw_phone}» (нужен номер из 10–11 цифр, например +7 916 123-45-67)")
    elif phone[-10:] not in re.sub(r"\D", "", client_text):
        invalid.append(f"телефон «{raw_phone}» — клиент его не называл, спроси номер")

    desired_time = str(args.get("desired_time") or "").strip()
    day = resolve_date(desired_time, date.today())
    if not desired_time:
        missing.append("желаемое время")
    elif (not (_numbers(desired_time) - {"0"}) <= _numbers(client_text)
          or any(not re.search(r"\b" + w, said) for w in DAY_WORD_RE.findall(_lower(desired_time)))):
        invalid.append(f"желаемое время «{desired_time}» — клиент его так не называл: спроси день и время "
                       f"и передай его словами клиента")
    elif day is None:
        invalid.append(f"желаемое время «{desired_time}» — нет дня: спроси у клиента день недели или дату")
    elif studio.contacts.schedule and studio.contacts.schedule.get(DAY_KEYS[day.weekday()]) is None:
        invalid.append(f"желаемое время: {date_text(day)} — выходной, предложи клиенту рабочий день")

    ids = re.findall(r"[^,;\s]+", str(args.get("service_id") or ""))  # несколько услуг — через запятую
    service_id = ",".join(ids) or None
    service = str(args.get("service") or "").strip()
    if ids:
        nodes = [studio.get_service(i) or studio.get_class(i) for i in ids]
        unknown = [i for i, n in zip(ids, nodes) if n is None]
        if unknown:
            invalid.append(f"service_id «{', '.join(unknown)}» — нет в каталоге")
            service_id = None
        else:
            service = "; ".join(n.name.replace(SEP, " — ") for n in nodes)
    if not service and not any(i.startswith("service_id") for i in invalid):
        missing.append("услуга")

    if missing or invalid:
        problem = {"error": "Заявка не создана: не хватает или неверны данные. Уточни у клиента и вызови create_booking снова."}
        if missing:
            problem["missing"] = missing
        if invalid:
            problem["invalid"] = invalid
        return None, problem

    comment = str(args.get("comment") or "").strip()[:300] or None
    if comment and _lower(comment).strip(" .") in _lower(service):  # модель повторила услугу
        comment = None
    now = datetime.now()
    booking = Booking(id=f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}", created_at=now.isoformat(timespec="seconds"),
                      source=source, name=name, phone=phone, service=service,
                      service_id=service_id, desired_time=desired_time[:100],
                      date=day.isoformat() if day else None, comment=comment)
    return booking, None


def _fields(b: Booking) -> dict:
    """Поля заявки для шаблонов: к услуге не из каталога — пометка, ко времени — вычисленная дата."""
    when = b.desired_time
    if b.date:
        when += f" ({date_text(date.fromisoformat(b.date))})"
    return dict(name=b.name, phone=display_phone(b.phone), desired_time=when,
                service=b.service + ("" if b.service_id else NOT_IN_CATALOG),
                comment=f"Комментарий: {b.comment}\n" if b.comment else "")


def same(a: Booking, b: Booking) -> bool:
    """Те же данные заявки (id и время создания не сравниваются)."""
    keys = ("name", "phone", "service", "service_id", "desired_time", "comment")
    return all(getattr(a, k) == getattr(b, k) for k in keys)


def confirmation_text(b: Booking) -> str:
    return CONFIRM_TEMPLATE.format(**_fields(b))


def saved_text(b: Booking) -> str:
    return SAVED_TEXT.format(name=b.name, phone=display_phone(b.phone))


def admin_text(b: Booking, contact: str | None = None) -> str:
    """Уведомление администратору о подтверждённой заявке. contact — как связаться в мессенджере."""
    return ADMIN_TEMPLATE.format(**_fields(b), id=b.id,
                                 contact=f"{contact}\n" if contact else "",
                                 created_at=b.created_at.replace("T", " "))


def parse_confirmation(text: str) -> bool | None:
    """Короткое «да» / «нет» -> True / False. Всё остальное (в т.ч. «нет, телефон другой: …») -> None:
    такое сообщение уходит модели как исправление."""
    words = re.sub(r"[^\w+\-]+", " ", text.lower().replace("ё", "е")).split()
    if not words or len(words) > 5 or any(ch.isdigit() for ch in text):
        return None
    if words[0] in NO_WORDS or NEGATIONS & set(words):
        return False if len(words) <= 3 else None  # длинное «нет, …» — это уже исправление
    if words[0] in YES_WORDS or "верно" in words or "правильно" in words:
        return True
    return None


def is_cancel(text: str) -> bool:
    """«Я передумал», «отмена», «запись не нужна» -> True. Длинные сообщения не разбираем — их получит модель."""
    t = _lower(text)
    words = re.sub(r"[^\w]+", " ", t).split()
    if not words or len(words) > 6 or any(ch.isdigit() for ch in t) or YES_WORDS & set(words) or "верно" in words:
        return False
    return bool(CANCEL_WORDS & set(words)) or any(p in t for p in CANCEL_PHRASES)


def save(b: Booking) -> None:
    """Дописать заявку в bookings.json (через временный файл, чтобы не испортить при сбое)."""
    with _file_lock:
        items = json.loads(BOOKINGS_FILE.read_text(encoding="utf-8")) if BOOKINGS_FILE.exists() else []
        items.append(asdict(b))
        tmp = BOOKINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, BOOKINGS_FILE)
