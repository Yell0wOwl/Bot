"""Диалог с ботом в консоли. Запуск: python run.py"""
from pathlib import Path

from agent import Agent
from kb import KnowledgeBase
from cloud_inference import chat  # локальная модель: from inference import chat
from schema import load_studio

STUDIO_FILE = Path(__file__).parent / "modnoe_mesto.json"
STOP_REQUEST = "exit"


def main():
    studio = load_studio(STUDIO_FILE)
    agent = Agent(KnowledgeBase(studio), chat)

    print(f"Бот студии «{studio.name}». Для завершения введите {STOP_REQUEST}\n")
    while True:
        text = input("Вы: ").strip()
        if text == STOP_REQUEST:
            break
        if text:
            print(f"Бот: {agent.ask(text)}\n")


if __name__ == "__main__":
    main()
