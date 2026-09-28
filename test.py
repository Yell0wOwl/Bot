"""Прогон бота на вопросах из test.json. Запуск: python test.py

Ответы бота записываются в test_results.json в том же формате, что и test.json.
В конце выводится среднее время ответа (вопросы, завершившиеся ошибкой, не учитываются).
Каждый вопрос задаётся с чистой историей диалога. Результаты сохраняются после
каждого вопроса, поэтому при остановке прогона готовая часть не теряется.
"""
import json
import statistics
import time
from pathlib import Path

from agent import Agent
from kb import KnowledgeBase
from cloud_inference import chat  # локальная модель: from inference import chat
from schema import load_studio

BASE_DIR = Path(__file__).parent
STUDIO_FILE = BASE_DIR / "modnoe_mesto.json"
TESTS_FILE = BASE_DIR / "test.json"
RESULTS_FILE = BASE_DIR / "test_results.json"


def main():
    kb = KnowledgeBase(load_studio(STUDIO_FILE))
    tests = json.loads(TESTS_FILE.read_text(encoding="utf-8"))
    results = []
    durations = []  # время ответа по каждому вопросу без ошибок, секунды
    errors = 0

    for i, test in enumerate(tests, 1):
        agent = Agent(kb, chat, name=f"test{test['id']}")  # новый агент — вопросы не влияют друг на друга
        start = time.monotonic()
        try:
            answer = agent.ask(test["question"])
            durations.append(time.monotonic() - start)
            took = f"{durations[-1]:.1f} с"
        except Exception as e:
            answer = f"ОШИБКА: {type(e).__name__}: {e}"
            errors += 1
            took = "ошибка"
        results.append({**test, "answer": answer})
        RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(tests)}] ({took}) {test['question']}\n  → {answer}\n")

    print(f"Готово, ответы записаны в {RESULTS_FILE.name}")
    if durations:
        print(f"Среднее время ответа: {statistics.mean(durations):.1f} с "
              f"(медиана {statistics.median(durations):.1f} с, "
              f"мин {min(durations):.1f} с, макс {max(durations):.1f} с; ответов: {len(durations)})")
    if errors:
        print(f"Ошибок: {errors} — они не учтены во времени")


if __name__ == "__main__":
    main()
