"""Формат базы знаний студии (JSON) и его проверка.

Запуск как скрипт — проверить файл:
    python schema.py example.json
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints

SEP = "/"  # разделитель уровней в name

Time = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
Day = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
NodeId = Annotated[str, StringConstraints(pattern=r"^\d+(\.\d+)*$")]


class Model(BaseModel):
    # лишние поля в JSON — ошибка: так опечатка в ключе ("servises") не пройдёт молча
    model_config = ConfigDict(extra="forbid")


# ---------- основная информация ----------

class Contacts(Model):
    address: str | None = None
    phone: list[str] = []
    email: list[str] = []
    # [открытие, закрытие]; null или отсутствие дня — выходной
    schedule: dict[Day, tuple[Time, Time] | None] = {}


class BodyClass(Model):
    code: Annotated[str, StringConstraints(pattern=r"^\d+$")]
    name: str


# ---------- дерево услуг ----------

class Modifier(Model):
    name: str
    type: Literal["percent", "fixed"]  # +N% или +N ₽
    value: float


class Service(Model):
    """Лист дерева: конкретная услуга для конкретного класса кузова."""
    id: NodeId
    name: str
    body_class: str | None = None  # None — цена одинакова для любого класса
    price_from: int | None = None  # price_from == price_to — точная цена
    price_to: int | None = None    # price_to is None — «от price_from»
    currency: str = "RUB"
    duration: str | None = None
    modifiers: list[Modifier] = []


class ServiceClass(Model):
    """Промежуточный узел дерева: категория или позиция прайса."""
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
    service_class_id: str | None = None
    service_class: str | None = None
    message: str


# ---------- студия целиком ----------

class Studio(Model):
    name: str
    source_url: str | None = None
    contacts: Contacts = Contacts()
    body_classes: list[BodyClass] = []
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
        """Все service_class в порядке обхода: (узел, родитель или None для корня)."""
        def walk(node: ServiceClass | ServiceTree, parent: ServiceClass | None):
            for c in node.service_classes:
                yield c, parent
                yield from walk(c, c)
        yield from walk(self.service_tree, None)

    def iter_services(self, root: ServiceClass | None = None) -> Iterator[tuple[Service, ServiceClass | None]]:
        """Все листья (услуга, её родитель). root — обойти только это поддерево."""
        def walk(node: ServiceClass | ServiceTree, parent: ServiceClass | None):
            for s in node.services:
                yield s, parent
            for c in node.service_classes:
                yield from walk(c, c)
        start = root if root is not None else self.service_tree
        yield from walk(start, root)

    # --- поиск по id ---

    def get_class(self, class_id: str) -> ServiceClass | None:
        return self._classes.get(class_id)

    def get_service(self, service_id: str) -> Service | None:
        return self._services.get(service_id)

    def parent_of(self, node_id: str) -> ServiceClass | None:
        return self._parent.get(node_id)

    def body_class_name(self, code: str | None) -> str | None:
        return next((b.name for b in self.body_classes if b.code == code), None)


def load_studio(path: str | Path) -> Studio:
    return Studio.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ---------- проверки содержимого ----------

def _fmt(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def check(studio: Studio) -> list[Warning]:
    """Проверки, которые не выразить типами: связность дерева, id, цены."""
    out: list[Warning] = []

    def add(level, msg, node: ServiceClass | None = None):
        out.append(Warning(level=level, message=msg,
                           service_class_id=node.id if node else None,
                           service_class=node.name if node else None))

    # классы кузова: коды 1..N подряд
    codes = [b.code for b in studio.body_classes]
    if codes != [str(i) for i in range(1, len(codes) + 1)]:
        add("error", f"Коды body_classes должны идти подряд с 1, сейчас: {codes}")
    code_set = set(codes)

    # уникальность id
    all_ids = [c.id for c, _ in studio.iter_classes()] + [s.id for s, _ in studio.iter_services()]
    for node_id, n in Counter(all_ids).items():
        if n > 1:
            add("error", f"id {node_id} встречается {n} раз")

    def check_child(node: ServiceClass | Service, parent: ServiceClass | None):
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

    for srv, parent in studio.iter_services():
        check_child(srv, parent)
        if srv.body_class is not None and srv.body_class not in code_set:
            add("error", f"Услуга {srv.id}: неизвестный body_class «{srv.body_class}»", parent)
        if srv.price_from is None and srv.price_to is None:
            add("info", f"Услуга {srv.id}: цена не указана", parent)
        if srv.price_from is not None and srv.price_to is not None and srv.price_from > srv.price_to:
            add("error", f"Услуга {srv.id}: price_from {_fmt(srv.price_from)} больше price_to {_fmt(srv.price_to)}", parent)
        for p in (srv.price_from, srv.price_to):
            if p is not None and p <= 0:
                add("error", f"Услуга {srv.id}: неположительная цена {p}", parent)

    # цены внутри одной позиции по классам кузова: не падают, без скачков > 3x
    for cls, _ in studio.iter_classes():
        by_class = sorted(((int(s.body_class), s.price_from) for s in cls.services
                           if s.body_class is not None and s.price_from is not None))
        for (c1, p1), (c2, p2) in zip(by_class, by_class[1:]):
            if p2 < p1:
                add("error", f"Цена {c2} класса ({_fmt(p2)}) ниже {c1} класса ({_fmt(p1)}) — похоже на опечатку", cls)
            elif p2 > 3 * p1:
                add("error", f"Скачок цены {c1} → {c2} класс ({_fmt(p1)} → {_fmt(p2)}) — похоже на опечатку", cls)

    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Использование: python schema.py <файл.json>")
    st = load_studio(sys.argv[1])
    n_cls = sum(1 for _ in st.iter_classes())
    n_srv = sum(1 for _ in st.iter_services())
    print(f"OK: «{st.name}», категорий {n_cls}, услуг {n_srv}")
    problems = check(st)
    for w in problems:
        where = f" [{w.service_class_id}] {w.service_class}" if w.service_class_id else ""
        print(f"{w.level.upper():7}{where}: {w.message}")
    if not problems:
        print("Замечаний нет")
