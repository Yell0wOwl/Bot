"""Прогон бота на диалогах из test.json. Запуск: python test.py

Каждый тест — диалог из 1–5 сообщений клиента. Сообщения задаются по очереди одному агенту,
каждый тест начинается с новым агентом (пустая история). Ответы бота записываются в
test_results.json в том же формате, что и test.json. Заявки, подтверждённые в тестах, сохраняются
в bookings.json как обычные — их поле source равно test<номер теста>.
Результаты сохраняются после каждого теста, поэтому при остановке прогона готовая часть не теряется.
"""
import json
import statistics
import time
from pathlib import Path

from agent import Agent
from kb import KnowledgeBase
from cloud_inference import chat
from schema import load_studio

BASE_DIR = Path(__file__).parent
STUDIO_FILE = BASE_DIR / "diesel_hard.json"
TESTS_FILE = BASE_DIR / "test.json"
RESULTS_FILE = BASE_DIR / "test_results.json"


def main():
    kb = KnowledgeBase(load_studio(STUDIO_FILE))
    tests = json.loads(TESTS_FILE.read_text(encoding="utf-8"))
    results = []
    durations = []  # время ответа на каждое сообщение без ошибок, секунды
    errors = 0
    saved_bookings = 0

    for i, test in enumerate(tests, 1):
        agent = Agent(kb, chat, name=f"test{test['id']}")  # новый агент — тесты не влияют друг на друга
        dialog = []
        print(f"[{i}/{len(tests)}] {test['category']}")
        for turn in test["dialog"]:
            start = time.monotonic()
            try:
                answer = agent.ask(turn["question"])
                durations.append(time.monotonic() - start)
                took = f"{durations[-1]:.1f} с"
            except Exception as e:
                answer = f"ОШИБКА: {type(e).__name__}: {e}"
                errors += 1
                took = "ошибка"
            dialog.append({"question": turn["question"], "answer": answer})
            print(f"  клиент: {turn['question']}\n  бот ({took}): {answer}\n")
            if answer.startswith("ОШИБКА"):
                break  # продолжать диалог после ошибки бессмысленно
        saved_bookings += len(agent.new_bookings)
        results.append({**test, "dialog": dialog})
        RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Готово, ответы записаны в {RESULTS_FILE.name}")
    if durations:
        print(f"Среднее время ответа: {statistics.mean(durations):.1f} с "
              f"(медиана {statistics.median(durations):.1f} с, "
              f"мин {min(durations):.1f} с, макс {max(durations):.1f} с; ответов: {len(durations)})")
    print(f"Заявок сохранено в bookings.json: {saved_bookings}")
    if errors:
        print(f"Ошибок: {errors} — они не учтены во времени")


if __name__ == "__main__":
    main()
