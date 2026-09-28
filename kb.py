"""База знаний студии: всё, что бот знает, и инструменты, которыми он это достаёт.

Цены форматирует код, а не модель: модель только пересказывает готовую строку.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from schema import SEP, Service, ServiceClass, Studio, check

DAYS_RU = {"mon": "пн", "tue": "вт", "wed": "ср", "thu": "чт", "fri": "пт", "sat": "сб", "sun": "вс"}
DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]  # по date.weekday()
WEEKDAYS_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
WEEKDAYS_WHEN_RU = ["в понедельник", "во вторник", "в среду", "в четверг", "в пятницу", "в субботу", "в воскресенье"]
MONTHS_RU = ["", "января", "февраля", "марта", "апреля", "мая", "июня", "июля",
             "августа", "сентября", "октября", "ноября", "декабря"]
RELATIVE_DAYS = {0: "сегодня", 1: "завтра", 2: "послезавтра"}
CALENDAR_DAYS = 7  # на сколько дней вперёд расписываем график
MAX_SERVICES = 60  # сколько услуг максимум отдаём модели за один вызов
STOPWORDS = {"сколько", "стоит", "стоимость", "цена", "цены", "почем", "почём", "у", "вас", "на",
             "для", "и", "в", "с", "по", "а", "мне", "нужно", "нужна", "хочу", "можно", "есть", "ли",
             "какая", "какие", "какой", "машины", "машину", "авто", "автомобиля", "автомобиль"}


def money(x: int) -> str:
    return f"{x:,}".replace(",", " ") + " ₽"


def price_text(s: Service) -> str:
    lo, hi = s.price_from, s.price_to
    if lo is None and hi is None:
        return "цена по запросу"
    if lo is None:
        return f"до {money(hi)}"
    if hi is None:
        return f"от {money(lo)}"
    if lo == hi:
        return money(lo)
    return f"от {money(lo)} до {money(hi)}"


def modifier_text(m) -> str:
    val = f"{m.value:g}%" if m.type == "percent" else money(int(m.value))
    return f"{m.name}: +{val}"


def last_segment(name: str) -> str:
    return name.rsplit(SEP, 1)[-1]


def _stems(text: str) -> list[str]:
    words = re.sub(r"[^\w]+", " ", text.lower().replace("ё", "е")).split()
    return [w[:4] for w in words if w not in STOPWORDS and len(w) > 1]


class KnowledgeBase:
    def __init__(self, studio: Studio):
        self.studio = studio
        # позиции с подозрительными ценами: бот их не называет
        problems = list(studio.warnings) + check(studio)
        self.flagged = {w.service_class_id for w in problems if w.level == "error" and w.service_class_id}

    # ---------- то, что кладётся в системный промпт ----------

    def studio_info(self) -> str:
        st, c = self.studio, self.studio.contacts
        lines = [f"Название: {st.name}"]
        if c.address:
            lines.append(f"Адрес: {c.address}")
        if c.phone:
            lines.append("Телефон: " + ", ".join(c.phone))
        if c.email:
            lines.append("Email: " + ", ".join(c.email))
        if c.schedule:
            days = [f"{DAYS_RU[d]} {v[0]}–{v[1]}" if v else f"{DAYS_RU[d]} выходной"
                    for d, v in c.schedule.items()]
            lines.append("График: " + ", ".join(days))
        if st.body_classes:
            lines.append("Классы кузова: " + ", ".join(b.name for b in st.body_classes))
        for q in st.faq:
            lines.append(f"- {q}")
        return "\n".join(lines)

    @staticmethod
    def now() -> datetime:
        """Текущее время по часам компьютера, на котором запущен бот."""
        return datetime.now()

    def _hours(self, day: date) -> tuple[str, str] | None:
        return self.studio.contacts.schedule.get(DAY_KEYS[day.weekday()])

    @staticmethod
    def _day_name(day: date, today: date) -> str:
        """«завтра, пятница 26 сентября» / «суббота 27 сентября»."""
        name = f"{WEEKDAYS_RU[day.weekday()]} {day.day} {MONTHS_RU[day.month]}"
        rel = RELATIVE_DAYS.get((day - today).days)
        return f"{rel}, {name}" if rel else name

    def calendar(self, now: datetime) -> str:
        """Текущий момент и график на ближайшие дни — готовым текстом, чтобы модель
        не вычисляла дни недели сама: маленькие модели в этом ошибаются."""
        today, t = now.date(), now.strftime("%H:%M")
        lines = [f"Сейчас {WEEKDAYS_RU[today.weekday()]}, {today.day} {MONTHS_RU[today.month]} "
                 f"{today.year} года, {t}."]
        if not self.studio.contacts.schedule:
            lines.append("График работы студии не указан — время работы пусть уточнят по телефону.")
            return "\n".join(lines)

        hours = self._hours(today)
        if hours and hours[0] <= t < hours[1]:
            lines.append(f"Студия сейчас открыта, работает до {hours[1]}.")
        elif hours and t < hours[0]:
            lines.append(f"Студия сейчас закрыта, сегодня откроется в {hours[0]}.")
        else:
            for i in range(1, CALENDAR_DAYS + 1):
                day = today + timedelta(days=i)
                if h := self._hours(day):
                    when = f"{WEEKDAYS_WHEN_RU[day.weekday()]} {day.day} {MONTHS_RU[day.month]}"
                    if rel := RELATIVE_DAYS.get(i):
                        when = f"{rel}, {when}"
                    lines.append(f"Студия сейчас закрыта. Откроется {when}, в {h[0]}.")
                    break
            else:
                lines.append("Студия сейчас закрыта.")

        lines.append("График на ближайшие дни:")
        for i in range(CALENDAR_DAYS):
            day = today + timedelta(days=i)
            h = self._hours(day)
            lines.append(f"- {self._day_name(day, today)}: {f'{h[0]}–{h[1]}' if h else 'выходной'}")
        return "\n".join(lines)

    def outline(self, max_depth: int = 10) -> str:
        """Каталог без цен: дерево категорий с id, по одному узлу на строку."""
        out = []

        def walk(node, depth):
            for c in node.service_classes:
                out.append("  " * depth + f"[{c.id}] {last_segment(c.name)}")
                if depth + 1 < max_depth:
                    walk(c, depth + 1)
            # услуги без разбивки по классам кузова тоже показываем в каталоге
            for s in node.services:
                if s.body_class is None:
                    out.append("  " * depth + f"[{s.id}] {last_segment(s.name)}")

        walk(self.studio.service_tree, 0)
        return "\n".join(out)

    # ---------- инструменты ----------

    def _service_row(self, s: Service, parent: ServiceClass | None,
                     category: ServiceClass | None = None) -> dict:
        row = {"id": s.id, "service": parent.name if parent and s.body_class else s.name,
               "body_class": self.studio.body_class_name(s.body_class) or "любой"}
        if category is not None:  # не повторяем имя категории в каждой строке
            rel = row["service"].removeprefix(category.name).lstrip(SEP)
            if rel:
                row["service"] = rel
            else:
                del row["service"]
        if parent is not None and parent.id in self.flagged:
            row["price"] = "НЕ НАЗЫВАТЬ: цена уточняется у администратора"
        else:
            row["price"] = price_text(s)
            if s.modifiers:
                row["surcharges"] = [modifier_text(m) for m in s.modifiers]
        if s.duration:
            row["duration"] = s.duration
        return row

    def get_prices(self, service_class_id: str, body_class: str | None = None) -> dict:
        """Все услуги внутри категории (рекурсивно), при желании — только для одного класса кузова."""
        cls = self.studio.get_class(service_class_id)
        if cls is None:
            srv = self.studio.get_service(service_class_id)
            if srv is None:
                return {"error": f"Нет категории с id {service_class_id}. Проверь id по каталогу или вызови search_services."}
            return {"services": [self._service_row(srv, self.studio.parent_of(srv.id))]}

        rows = [self._service_row(s, p, cls) for s, p in self.studio.iter_services(cls)
                if body_class is None or s.body_class in (None, body_class)]
        result: dict = {"category": cls.name}
        if len(rows) > MAX_SERVICES:
            result["note"] = (f"Услуг слишком много ({len(rows)}). Выбери подкатегорию по каталогу "
                              f"и вызови get_prices для неё.")
            result["subcategories"] = [f"[{c.id}] {last_segment(c.name)}" for c in cls.service_classes]
            return result
        result["services"] = rows
        # диапазон по всем классам — готовой строкой, чтобы модели не пришлось считать
        if body_class is None and cls.id not in self.flagged and cls.services and not cls.service_classes:
            lows = [s.price_from for s in cls.services if s.price_from is not None]
            if len(set(lows)) > 1:
                result["range_all_classes"] = f"от {money(min(lows))} до {money(max(lows))} в зависимости от класса кузова"
            elif lows and len(cls.services) > 1:
                result["range_all_classes"] = f"{price_text(cls.services[0])} для любого класса кузова"
        return result

    def search_services(self, query: str, limit: int = 8) -> dict:
        """Поиск категорий по словам запроса (по префиксам слов, без морфологии)."""
        q = set(_stems(query))
        if not q:
            return {"results": [], "note": "Пустой запрос"}
        scored = []
        for cls, _ in self.studio.iter_classes():
            if not cls.services:  # ищем среди позиций прайса, а не групп
                continue
            hay = _stems(cls.name)
            hits = sum(1 for w in q if any(h.startswith(w) or w.startswith(h) for h in hay))
            if hits:
                scored.append((hits / len(q), -len(cls.name), cls))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return {"results": [f"[{c.id}] {c.name}" for _, _, c in scored[:limit]]}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_prices",
            "description": "Цены на услуги из категории каталога (со всеми вложенными услугами). "
                           "Вызывай перед тем, как назвать любую цену.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_class_id": {"type": "string", "description": "id категории из каталога, например \"6.1.15\""},
                    "body_class": {"type": "string", "description": "Код класса кузова, если он точно известен. Иначе не передавай."},
                },
                "required": ["service_class_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_services",
            "description": "Поиск услуг по словам, если нужную категорию не удалось найти в каталоге.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Ключевые слова, например \"керамика кузов\""}},
                "required": ["query"],
            },
        },
    },
]
