"""База знаний компании: всё, что бот знает, и инструменты, которыми он это достаёт.

Цены подбирает и считает код, а не модель: модель получает готовую строку и пересказывает её.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from schema import SEP, NumRange, Parameter, Price, Service, ServiceClass, Studio

DAYS_RU = {"mon": "пн", "tue": "вт", "wed": "ср", "thu": "чт", "fri": "пт", "sat": "сб", "sun": "вс"}
DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]  # по date.weekday()
WEEKDAYS_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
WEEKDAYS_WHEN_RU = ["в понедельник", "во вторник", "в среду", "в четверг", "в пятницу", "в субботу", "в воскресенье"]
MONTHS_RU = ["", "января", "февраля", "марта", "апреля", "мая", "июня", "июля",
             "августа", "сентября", "октября", "ноября", "декабря"]
RELATIVE_DAYS = {0: "сегодня", 1: "завтра", 2: "послезавтра"}
CALENDAR_DAYS = 7   # на сколько дней вперёд расписываем график
MAX_SERVICES = 60   # сколько услуг максимум отдаём модели за один вызов
ASK_ADMIN = "УТОЧНИТЬ У АДМИНИСТРАТОРА"
STOPWORDS = {"сколько", "стоит", "стоимость", "цена", "цены", "почем", "почём", "у", "вас", "на",
             "для", "и", "в", "с", "по", "а", "мне", "нужно", "нужна", "хочу", "можно", "есть", "ли",
             "какая", "какие", "какой", "машины", "машину", "авто", "автомобиля", "автомобиль"}


def money(x: float) -> str:
    return f"{round(x):,}".replace(",", " ") + " ₽"


def last_segment(name: str) -> str:
    return name.rsplit(SEP, 1)[-1]


def _stems(text: str) -> list[str]:
    words = re.sub(r"[^\w]+", " ", text.lower().replace("ё", "е")).split()
    return [w[:4] for w in words if w not in STOPWORDS and len(w) > 1]


def _num(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


class KnowledgeBase:
    def __init__(self, studio: Studio):
        self.studio = studio

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
        for q in st.faq:
            lines.append(f"- {q}")
        return "\n".join(lines)

    def parameters_info(self) -> str:
        """Параметры, от которых зависит цена, — для промпта: что можно передавать в get_prices."""
        if not self.studio.parameters:
            return "Цены не зависят от параметров автомобиля — параметры в get_prices не передавай."
        lines = []
        for p in self.studio.parameters:
            if p.type == "choice":
                vals = []
                for v in p.values:
                    ex = f" (например: {', '.join(v.examples)})" if v.examples else ""
                    vals.append(f"«{v.code}» — {v.name}{ex}")
                line = f"- {p.id} — {p.name}: " + "; ".join(vals)
            else:
                line = f"- {p.id} — {p.name}: число" + (f", {p.unit}" if p.unit else "")
            if p.default is not None:
                line += f". Если клиент не сказал — считается «{p.value_name(p.default)}»"
            lines.append(line + ".")
        return "\n".join(lines)

    def outline(self) -> str:
        """Каталог без цен: дерево категорий и услуг с id, по одному узлу на строку."""
        out = []

        def walk(node, depth):
            for s in node.services:
                out.append("  " * depth + f"[{s.id}] {last_segment(s.name)}")
            for c in node.service_classes:
                out.append("  " * depth + f"[{c.id}] {last_segment(c.name)}")
                walk(c, depth + 1)

        walk(self.studio.service_tree, 0)
        return "\n".join(out)

    # ---------- календарь ----------

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
        не вычисляла дни недели сама."""
        today, t = now.date(), now.strftime("%H:%M")
        lines = [f"Сейчас {WEEKDAYS_RU[today.weekday()]}, {today.day} {MONTHS_RU[today.month]} "
                 f"{today.year} года, {t}."]
        if not self.studio.contacts.schedule:
            lines.append("График работы не указан — время работы пусть уточнят по телефону.")
            return "\n".join(lines)

        hours = self._hours(today)
        if hours and hours[0] <= t < hours[1]:
            lines.append(f"Сейчас открыто, работаем до {hours[1]}.")
        elif hours and t < hours[0]:
            lines.append(f"Сейчас закрыто, сегодня откроемся в {hours[0]}.")
        else:
            for i in range(1, CALENDAR_DAYS + 1):
                day = today + timedelta(days=i)
                if h := self._hours(day):
                    when = f"{WEEKDAYS_WHEN_RU[day.weekday()]} {day.day} {MONTHS_RU[day.month]}"
                    if rel := RELATIVE_DAYS.get(i):
                        when = f"{rel}, {when}"
                    lines.append(f"Сейчас закрыто. Откроемся {when}, в {h[0]}.")
                    break
            else:
                lines.append("Сейчас закрыто.")

        lines.append("График на ближайшие дни:")
        for i in range(CALENDAR_DAYS):
            day = today + timedelta(days=i)
            h = self._hours(day)
            lines.append(f"- {self._day_name(day, today)}: {f'{h[0]}–{h[1]}' if h else 'выходной'}")
        return "\n".join(lines)

    # ---------- подбор цены ----------

    def _clean_params(self, params: dict | None) -> tuple[dict, list[str]]:
        """Параметры от модели -> проверенные значения. Непонятное отбрасывается с пометкой."""
        clean, ignored = {}, []
        for pid, value in (params or {}).items():
            p = self.studio.get_param(pid)
            if p is None or value is None or value == "":
                ignored.append(f"{pid}={value}")
                continue
            if p.type == "choice":
                s = str(value).strip().lower()
                v = next((v for v in p.values if s in (v.code.lower(), v.name.lower())), None)
                if v is None:
                    ignored.append(f"{pid}={value}")
                else:
                    clean[pid] = v.code
            else:
                try:
                    clean[pid] = float(str(value).replace(",", "."))
                except ValueError:
                    ignored.append(f"{pid}={value}")
        return clean, ignored

    @staticmethod
    def _matches(param: Parameter, cond, value) -> bool:
        if param.type == "choice":
            return value in ([cond] if isinstance(cond, str) else cond)
        if isinstance(cond, NumRange):
            return (cond.min is None or value >= cond.min) and (cond.max is None or value <= cond.max)
        return value == float(cond)

    def _eval_when(self, when: dict, known: dict) -> tuple[bool, set[str]]:
        """(не противоречит ли известным значениям, какие параметры условия ещё неизвестны)."""
        unknown = set()
        for pid, cond in when.items():
            if pid in known:
                if not self._matches(self.studio.get_param(pid), cond, known[pid]):
                    return False, set()
            else:
                unknown.add(pid)
        return True, unknown

    def describe_when(self, when: dict) -> str:
        parts = []
        for pid, cond in when.items():
            p = self.studio.get_param(pid)
            if p.type == "choice":
                names = [p.value_name(c) for c in ([cond] if isinstance(cond, str) else cond)]
                parts.append(f"{p.name.lower()} — {' или '.join(names)}")
            elif isinstance(cond, NumRange):
                rng = (f"от {_num(cond.min)} " if cond.min is not None else "") + (f"до {_num(cond.max)}" if cond.max is not None else "")
                parts.append(f"{p.name.lower()} {rng.strip()}" + (f" {p.unit}" if p.unit else ""))
            else:
                parts.append(f"{p.name.lower()} {_num(cond)}" + (f" {p.unit}" if p.unit else ""))
        return ", ".join(parts) or "всегда"

    def price_for(self, srv: Service, client: dict) -> dict:
        """Цена услуги при известных параметрах клиента: готовые строки для модели."""
        known = dict(client)
        defaulted = {}
        for p in self.studio.parameters:
            if p.id not in known and p.default is not None:
                known[p.id] = p.default
                defaulted[p.id] = p.default

        # базовые цены и «к администратору», не противоречащие известному
        cands = []
        for rule in srv.prices:
            if rule.kind != "surcharge":
                ok, unknown = self._eval_when(rule.when, known)
                if ok:
                    cands.append((rule, unknown))
        row: dict = {}
        if not cands:
            row["price"] = ASK_ADMIN
            return row

        if not any(unknown for _, unknown in cands):
            # всё известно — берём самое конкретное правило
            best = max(len(r.when) for r, _ in cands)
            chosen = [r for r, _ in cands if len(r.when) == best]
            depends: set[str] = set()
            if any(r.kind == "ask_admin" for r in chosen):
                row["price"] = ASK_ADMIN
                return row
        else:
            chosen = [r for r, _ in cands]
            depends = set().union(*(unknown for _, unknown in cands))

        bases = [r for r in chosen if r.kind == "base"]
        admins = [r for r in chosen if r.kind == "ask_admin"]
        if not bases:
            row["price"] = ASK_ADMIN
            return row

        lo = min(r.from_ for r in bases)
        hi = None if any(r.to is None for r in bases) else max(r.to for r in bases)

        # надбавки: сказанное клиентом — применяем; зависящее от несказанного — текстом
        options = []
        for rule in srv.prices:
            if rule.kind != "surcharge":
                continue
            ok, unknown_client = self._eval_when(rule.when, client)
            if not ok:
                continue
            ok_eff, unknown_eff = self._eval_when(rule.when, known)
            if not unknown_client or (ok_eff and not unknown_eff):
                lo, hi = self._apply(rule, lo), (None if hi is None else self._apply(rule, hi))
            else:
                amount = f"+{_num(rule.add_percent)}%" if rule.add_percent is not None else f"+{money(rule.add)}"
                options.append(f"{self.describe_when(rule.when)}: {amount}")

        if hi is None:
            row["price"] = f"от {money(lo)}"
        elif round(lo) == round(hi):
            row["price"] = money(lo)
        else:
            row["price"] = f"от {money(lo)} до {money(hi)}"
        if srv.unit:
            row["price"] += f" {srv.unit}"

        distinct = {(r.from_, r.to) for r in bases}
        if depends and (len(distinct) > 1 or admins):
            row["depends_on"] = [{"parameter": self.studio.get_param(pid).name,
                                  "question": self.studio.get_param(pid).question} for pid in sorted(depends)]
        if admins:
            row["admin_for"] = "; ".join(f"при «{self.describe_when(r.when)}» цену уточнит администратор" for r in admins)
        if options:
            row["surcharges"] = options
            assumed = [f"{self.studio.get_param(pid).name.lower()} — {self.studio.get_param(pid).value_name(v)}"
                       for pid, v in defaulted.items() if any(pid in r.when for r in srv.prices if r.kind == "surcharge")]
            if assumed:
                row["assumed"] = "цена указана для: " + ", ".join(assumed)
        return row

    @staticmethod
    def _apply(rule: Price, x: float) -> float:
        return x * (1 + rule.add_percent / 100) if rule.add_percent is not None else x + rule.add

    # ---------- инструменты ----------

    def get_prices(self, id: str, params: dict | None = None) -> dict:
        """Цены услуги или всех услуг категории при известных параметрах клиента."""
        client, ignored = self._clean_params(params)
        srv = self.studio.get_service(id)
        cls = self.studio.get_class(id)
        if srv is None and cls is None:
            return {"error": f"Нет услуги или категории с id {id}. Проверь id по каталогу или вызови search_services."}

        def row(s: Service, base: str) -> dict:
            r = {"id": s.id, "service": s.name.removeprefix(base).lstrip(SEP) or last_segment(s.name)}
            r.update(self.price_for(s, client))
            if s.duration:
                r["duration"] = s.duration
            return r

        result: dict = {}
        if srv is not None:
            parent = self.studio.parent_of(srv.id)
            result["services"] = [row(srv, parent.name if parent else "")]
        else:
            services = [s for s, _ in self.studio.iter_services(cls)]
            result["category"] = cls.name
            if len(services) > MAX_SERVICES:
                result["note"] = (f"Услуг слишком много ({len(services)}). Выбери подкатегорию по каталогу "
                                  f"и вызови get_prices для неё.")
                result["subcategories"] = [f"[{c.id}] {last_segment(c.name)}" for c in cls.service_classes]
                return result
            result["services"] = [row(s, cls.name) for s in services]
        if client:
            result["params_used"] = self.describe_when(client)
        if ignored:
            result["params_ignored"] = ignored
        return result

    def search_services(self, query: str, limit: int = 8) -> dict:
        """Поиск услуг по словам запроса (по первым 4 буквам слов, без морфологии)."""
        q = set(_stems(query))
        if not q:
            return {"results": [], "note": "Пустой запрос"}
        scored = []
        for srv, _ in self.studio.iter_services():
            hay = _stems(srv.name)
            hits = sum(1 for w in q if any(h.startswith(w) or w.startswith(h) for h in hay))
            if hits:
                scored.append((hits / len(q), -len(srv.name), srv))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return {"results": [f"[{s.id}] {s.name}" for _, _, s in scored[:limit]]}

    def tools(self) -> list[dict]:
        """Описания инструментов для модели. Параметры get_prices строятся из параметров компании."""
        props: dict = {"id": {"type": "string", "description": "id услуги или категории из каталога, например \"6.1.15\""}}
        if self.studio.parameters:
            pprops = {}
            for p in self.studio.parameters:
                if p.type == "choice":
                    pprops[p.id] = {"type": "string", "enum": [v.code for v in p.values],
                                    "description": f"{p.name}: " + "; ".join(f"{v.code} — {v.name}" for v in p.values)}
                else:
                    pprops[p.id] = {"type": "number", "description": p.name + (f", {p.unit}" if p.unit else "")}
            props["params"] = {"type": "object", "properties": pprops,
                               "description": "Параметры автомобиля клиента — только те, что он назвал. Не угадывай."}
        return [
            {"type": "function", "function": {
                "name": "get_prices",
                "description": "Цены услуги или всех услуг категории. Вызывай перед тем, как назвать любую цену.",
                "parameters": {"type": "object", "properties": props, "required": ["id"]}}},
            {"type": "function", "function": {
                "name": "search_services",
                "description": "Поиск услуг по словам, если нужную услугу не удалось найти в каталоге.",
                "parameters": {"type": "object",
                               "properties": {"query": {"type": "string", "description": "Ключевые слова, например \"керамика кузов\""}},
                               "required": ["query"]}}},
        ]
