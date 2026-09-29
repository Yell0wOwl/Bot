"""Формат базы знаний (JSON) и его проверка.

Запуск как скрипт — проверить файл:
    python schema.py modnoe_mesto.json

Устройство:
  parameters   — свойства машины клиента, от которых зависит цена (класс кузова, тип плёнки,
                 радиус дисков...). Могут отсутствовать: тогда у услуг одна цена или «к администратору».
  service_tree — дерево категорий; листья — услуги.
  service.prices — правила цены. Каждое действует при условии when и бывает трёх видов:
      базовая цена       {"when": {...}, "from": 28000, "to": 28000}   (to = null — «от»)
      надбавка           {"when": {...}, "add_percent": 10}  или  {"when": {...}, "add": 5000}
      к администратору   {"when": {...}, "ask_admin": true}              — цену бот не называет
  when: {} — всегда; {"body_class": "3"}; {"body_class": ["1", "2"]}; {"wheel_radius": {"min": 18, "max": 21}}.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Iterator, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints, model_validator

SEP = "/"  # разделитель уровней в name

Time = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
Day = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
NodeId = Annotated[str, StringConstraints(pattern=r"^\d+(\.\d+)*$")]
ParamId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class Model(BaseModel):
    # лишние поля в JSON — ошибка: так опечатка в ключе ("servises") не пройдёт молча
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------- основная информация ----------

class Contacts(Model):
    address: str | None = None
    phone: list[str] = []
    email: list[str] = []
    # [открытие, закрытие]; null или отсутствие дня — выходной
    schedule: dict[Day, tuple[Time, Time] | None] = {}


# ---------- параметры цены ----------

class ParamValue(Model):
    code: str
    name: str
    examples: list[str] = []  # например, модели машин этого класса кузова


class Parameter(Model):
    id: ParamId
    name: str
    question: str                        # как бот уточняет параметр у клиента
    type: Literal["choice", "number"]
    values: list[ParamValue] = []        # только для choice
    default: str | float | None = None   # значение, если клиент ничего не сказал
    ordered: bool = False                # choice: значения по возрастанию цены (для проверки)
    unit: str | None = None              # number: единица измерения

    def value_name(self, code) -> str:
        v = next((v for v in self.values if v.code == code), None)
        return v.name if v else str(code)


class NumRange(Model):
    min: float | None = None
    max: float | None = None


Condition = Union[str, list[str], NumRange, float]


class Price(Model):
    """Правило цены: базовая цена, надбавка или «к администратору»."""
    when: dict[str, Condition] = {}
    from_: int | None = Field(None, alias="from")
    to: int | None = None
    add_percent: float | None = None
    add: int | None = None
    ask_admin: bool = False

    @model_validator(mode="after")
    def _one_kind(self):
        kinds = [self.from_ is not None, self.add_percent is not None or self.add is not None, self.ask_admin]
        if sum(kinds) != 1:
            raise ValueError("правило цены должно быть ровно одного вида: from/to, add_percent/add или ask_admin")
        if self.to is not None and self.from_ is None:
            raise ValueError("to без from")
        if self.add_percent is not None and self.add is not None:
            raise ValueError("нельзя одновременно add_percent и add")
        return self

    @property
    def kind(self) -> Literal["base", "surcharge", "ask_admin"]:
        if self.ask_admin:
            return "ask_admin"
        return "base" if self.from_ is not None else "surcharge"


# ---------- дерево услуг ----------

class Service(Model):
    """Лист дерева: услуга (позиция прайса) со своими правилами цены."""
    id: NodeId
    name: str
    prices: list[Price] = Field(min_length=1)
    unit: str | None = None       # «за шт.», «за 4 шт.», «за нормо-час»
    currency: str = "RUB"
    duration: str | None = None


class ServiceClass(Model):
    """Промежуточный узел дерева: категория."""
    id: NodeId
    name: str
    service_classes: list[ServiceClass] = []
    services: list[Service] = []


class ServiceTree(Model):
    """Корень дерева: верхнеуровневые категории и услуги без категории."""
    service_classes: list[ServiceClass] = []
    services: list[Service] = []


class Warning(Model):
    level: Literal["error", "warning", "info"]
    service_class_id: str | None = None   # id категории или услуги, к которой относится замечание
    service_class: str | None = None
    message: str


# ---------- компания целиком ----------

class Studio(Model):
    name: str
    source_url: str | None = None
    contacts: Contacts = Contacts()
    parameters: list[Parameter] = []
    service_tree: ServiceTree = ServiceTree()
    faq: list[str] = []
    warnings: list[Warning] = []

    _classes: dict[str, ServiceClass] = PrivateAttr(default_factory=dict)
    _services: dict[str, Service] = PrivateAttr(default_factory=dict)
    _parent: dict[str, ServiceClass | None] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context) -> None:
        for cls, parent in self.iter_classes():
            self._classes[cls.id] = cls
            self._parent[cls.id] = parent
        for srv, parent in self.iter_services():
            self._services[srv.id] = srv
            self._parent[srv.id] = parent

    # --- обход дерева ---

    def iter_classes(self) -> Iterator[tuple[ServiceClass, ServiceClass | None]]:
        """Все категории в порядке обхода: (узел, родитель или None для корня)."""
        def walk(node, parent):
            for c in node.service_classes:
                yield c, parent
                yield from walk(c, c)
        yield from walk(self.service_tree, None)

    def iter_services(self, root: ServiceClass | None = None) -> Iterator[tuple[Service, ServiceClass | None]]:
        """Все услуги (услуга, её родитель). root — обойти только это поддерево."""
        def walk(node, parent):
            for s in node.services:
                yield s, parent
            for c in node.service_classes:
                yield from walk(c, c)
        yield from walk(root if root is not None else self.service_tree, root)

    # --- поиск по id ---

    def get_class(self, class_id: str) -> ServiceClass | None:
        return self._classes.get(class_id)

    def get_service(self, service_id: str) -> Service | None:
        return self._services.get(service_id)

    def parent_of(self, node_id: str) -> ServiceClass | None:
        return self._parent.get(node_id)

    def get_param(self, param_id: str) -> Parameter | None:
        return next((p for p in self.parameters if p.id == param_id), None)


def load_studio(path: str | Path) -> Studio:
    return Studio.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ---------- проверки содержимого ----------

def _fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def check(studio: Studio) -> list[Warning]:
    """Проверки, которые не выразить типами: связность дерева, параметры, цены."""
    out: list[Warning] = []

    def add(level, msg, node=None):
        out.append(Warning(level=level, message=msg,
                           service_class_id=node.id if node else None,
                           service_class=node.name if node else None))

    # параметры
    for pid, n in Counter(p.id for p in studio.parameters).items():
        if n > 1:
            add("error", f"Параметр {pid} объявлен {n} раз")
    for p in studio.parameters:
        codes = [v.code for v in p.values]
        if p.type == "choice":
            if not codes:
                add("error", f"Параметр {p.id}: у choice нет values")
            if len(set(codes)) != len(codes):
                add("error", f"Параметр {p.id}: повторяются коды значений")
            if p.default is not None and p.default not in codes:
                add("error", f"Параметр {p.id}: default «{p.default}» нет среди values")
        else:
            if codes:
                add("error", f"Параметр {p.id}: у number не бывает values")
            if p.default is not None and not isinstance(p.default, (int, float)):
                add("error", f"Параметр {p.id}: default должен быть числом")

    # уникальность id и связность дерева
    all_ids = [c.id for c, _ in studio.iter_classes()] + [s.id for s, _ in studio.iter_services()]
    for node_id, n in Counter(all_ids).items():
        if n > 1:
            add("error", f"id {node_id} встречается {n} раз")

    def check_child(node, parent):
        if parent is None:
            if "." in node.id:
                add("error", f"Узел верхнего уровня «{node.name}» имеет составной id {node.id}")
            if SEP in node.name:
                add("error", f"Узел верхнего уровня «{node.name}» содержит «{SEP}» в имени")
        else:
            if not node.id.startswith(parent.id + ".") or node.id.count(".") != parent.id.count(".") + 1:
                add("error", f"id {node.id} не продолжает id родителя {parent.id}", parent)
            last = node.name.removeprefix(parent.name + SEP)
            if last == node.name or not last or SEP in last:
                add("error", f"Имя «{node.name}» должно быть «{parent.name}{SEP}<имя>» без «{SEP}» в <имя>", parent)

    for cls, parent in studio.iter_classes():
        check_child(cls, parent)
        if not cls.service_classes and not cls.services:
            add("warning", "Пустая категория: нет ни подкатегорий, ни услуг", cls)

    # цены
    for srv, parent in studio.iter_services():
        check_child(srv, parent)
        _check_prices(studio, srv, add)
    return out


def _check_prices(studio: Studio, srv: Service, add) -> None:
    if not any(p.kind in ("base", "ask_admin") for p in srv.prices):
        add("error", "Только надбавки: нет ни базовой цены, ни ask_admin", srv)

    for rule in srv.prices:
        if rule.from_ is not None and rule.to is not None and rule.from_ > rule.to:
            add("error", f"from {_fmt(rule.from_)} больше to {_fmt(rule.to)}", srv)
        for x in (rule.from_, rule.to, rule.add):
            if x is not None and x <= 0:
                add("error", f"Неположительная цена {x}", srv)
        for pid, cond in rule.when.items():
            param = studio.get_param(pid)
            if param is None:
                add("error", f"В when неизвестный параметр «{pid}»", srv)
                continue
            if param.type == "choice":
                codes = {v.code for v in param.values}
                vals = [cond] if isinstance(cond, str) else cond if isinstance(cond, list) else None
                if vals is None:
                    add("error", f"Параметр {pid} — choice, в when нужен код или список кодов", srv)
                elif bad := [v for v in vals if v not in codes]:
                    add("error", f"Параметр {pid}: неизвестные значения {bad}", srv)
            elif not isinstance(cond, (NumRange, float, int)):
                add("error", f"Параметр {pid} — number, в when нужно число или {{min, max}}", srv)

    # цена по упорядоченному параметру (класс кузова): не падает, без скачков > 3x
    for param in studio.parameters:
        if param.type != "choice" or not param.ordered:
            continue
        order = {v.code: i for i, v in enumerate(param.values)}
        points = []  # (позиция значения, цена) по базовым ценам, где when — только этот параметр
        for rule in srv.prices:
            if rule.kind == "base" and set(rule.when) == {param.id}:
                cond = rule.when[param.id]
                for code in ([cond] if isinstance(cond, str) else cond):
                    if code in order:
                        points.append((order[code], code, rule.from_))
        points.sort()
        for (_, c1, p1), (_, c2, p2) in zip(points, points[1:]):
            if p2 < p1:
                add("error", f"Цена при {param.name.lower()} «{param.value_name(c2)}» ({_fmt(p2)}) ниже, "
                             f"чем при «{param.value_name(c1)}» ({_fmt(p1)}) — похоже на опечатку", srv)
            elif p2 > 3 * p1:
                add("error", f"Скачок цены «{param.value_name(c1)}» → «{param.value_name(c2)}» "
                             f"({_fmt(p1)} → {_fmt(p2)}) — похоже на опечатку", srv)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Использование: python schema.py <файл.json>")
    st = load_studio(sys.argv[1])
    n_cls = sum(1 for _ in st.iter_classes())
    n_srv = sum(1 for _ in st.iter_services())
    print(f"OK: «{st.name}», параметров {len(st.parameters)}, категорий {n_cls}, услуг {n_srv}")
    problems = check(st)
    for w in problems:
        where = f" [{w.service_class_id}] {w.service_class}" if w.service_class_id else ""
        print(f"{w.level.upper():7}{where}: {w.message}")
    if not problems:
        print("Замечаний нет")
