"""Инференс в облаке: Google Gemini (Interactions API).

Реализует контракт модели из agent.py:
    chat(messages, tools) -> Reply
Формат messages, tools и Reply описан в agent.py. Ключ и модель — в cloud_inference.cfg.
"""
import configparser
import sys
import time
from pathlib import Path

from google import genai

from agent import Reply, ToolCall

CONFIG_FILE = Path(__file__).parent / "cloud_inference.cfg"
RETRY_STATUSES = {429, 500, 502, 503, 504}  # перегрузка или временный сбой — стоит повторить
RETRY_DELAYS = [2, 5, 10]                   # паузы между повторами, секунды
MAX_CACHED_TURNS = 2000                     # сколько ходов модели помнить (см. _model_steps)

# ---------- настройки ----------

config = configparser.ConfigParser(interpolation=None)
if not config.read(CONFIG_FILE, encoding="utf-8"):
    sys.exit(f"Не найден файл настроек {CONFIG_FILE}")
gemini_cfg = config["gemini"]

API_KEY = gemini_cfg.get("api_key", "").strip()
if not API_KEY:
    sys.exit("В cloud_inference.cfg не задан api_key. Получите ключ: https://aistudio.google.com/apikey")
MODEL = gemini_cfg.get("model", "gemini-3.8-flash").strip()
THINKING_LEVEL = gemini_cfg.get("thinking_level", "low").strip()
FORCE_TOOL_CALL_FIRST = gemini_cfg.getboolean("force_tool_call_first", fallback=True)
MAX_OUTPUT_TOKENS = gemini_cfg.getint("max_output_tokens", fallback=2048)

client = genai.Client(api_key=API_KEY)

# Gemini «думает» перед ответом, и шаги размышлений (с подписью) нужно отправлять обратно
# в следующих запросах ровно такими, какими они пришли. В истории агента их нет, поэтому
# ходы модели хранятся здесь целиком: по id вызова инструмента или по тексту ответа.
_model_steps: dict[str, list[dict]] = {}


# ---------- контракт ----------

def chat(messages: list[dict], tools: list[dict]) -> Reply:
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    history = messages[1:] if system is not None else messages

    generation_config = {"thinking_level": THINKING_LEVEL, "max_output_tokens": MAX_OUTPUT_TOKENS}
    # последнее сообщение от клиента — первый шаг ответа: заставляем вызвать инструмент
    if FORCE_TOOL_CALL_FIRST and tools and messages[-1]["role"] == "user":
        generation_config["tool_choice"] = "any"

    request = {
        "model": MODEL,
        "input": _to_steps(history),
        "generation_config": generation_config,
        "store": False,  # историю ведёт агент, на сервере ничего не храним
    }
    if system:
        request["system_instruction"] = system
    if tools:
        request["tools"] = [_to_gemini_tool(t) for t in tools]

    return _to_reply(_create(request))


# ---------- история агента -> шаги Gemini ----------

def _text_step(step_type: str, text: str) -> dict:
    return {"type": step_type, "content": [{"type": "text", "text": text}]}


def _to_gemini_tool(tool: dict) -> dict:
    """{"type": "function", "function": {...}} -> формат Gemini: поля функции на верхнем уровне."""
    return {"type": "function", **tool["function"]}


def _to_steps(history: list[dict]) -> list[dict]:
    steps = []
    for m in history:
        role = m["role"]
        if role == "user":
            steps.append(_text_step("user_input", m["content"]))
        elif role == "system":  # замечание агента посреди диалога (проверка чисел)
            steps.append(_text_step("user_input", f"[Системное замечание] {m['content']}"))
        elif role == "assistant" and m.get("tool_calls"):
            cached = _model_steps.get(m["tool_calls"][0]["id"])
            if cached:
                steps.extend(cached)
            else:
                if m.get("content"):
                    steps.append(_text_step("model_output", m["content"]))
                steps.extend({"type": "function_call", "id": c["id"], "name": c["name"],
                              "arguments": c["arguments"]} for c in m["tool_calls"])
        elif role == "assistant":
            steps.extend(_model_steps.get(_text_key(m["content"])) or
                         [_text_step("model_output", m["content"])])
        elif role == "tool":
            step = {"type": "function_result", "call_id": m["tool_call_id"], "result": m["content"]}
            if m.get("name"):
                step["name"] = m["name"]
            steps.append(step)
    return steps


# ---------- ответ Gemini -> Reply ----------

def _to_reply(interaction) -> Reply:
    # новые шаги модели — всё после последнего входного шага (сообщения клиента или результата инструмента)
    new_steps = []
    for step in reversed(interaction.steps or []):
        if step.type in ("user_input", "function_result"):
            break
        new_steps.append(step)
    new_steps.reverse()

    text = "".join(c.text for s in new_steps if s.type == "model_output"
                   for c in (s.content or []) if getattr(c, "type", None) == "text").strip()
    calls = [ToolCall(id=s.id, name=s.name, arguments=dict(s.arguments or {}))
             for s in new_steps if s.type == "function_call"]

    if not text and not calls:
        raise RuntimeError(f"Gemini не вернул ни текста, ни вызова инструмента (статус {interaction.status})")

    raw = [s.model_dump(mode="json", by_alias=True, exclude_none=True) for s in new_steps]
    if calls:
        for c in calls:
            _remember(c.id, raw)
    else:
        _remember(_text_key(text), raw)
    return Reply(content=text, tool_calls=calls)


def _text_key(text: str) -> str:
    return "text:" + text.strip()


def _remember(key: str, raw_steps: list[dict]) -> None:
    _model_steps[key] = raw_steps
    while len(_model_steps) > MAX_CACHED_TURNS:  # старые ходы уже вытеснены из истории агента
        _model_steps.pop(next(iter(_model_steps)))


def _create(request: dict):
    """Запрос к Gemini с повтором при перегрузке и временных сбоях."""
    for delay in [*RETRY_DELAYS, None]:
        try:
            return client.interactions.create(**request)
        except Exception as e:
            if delay is None or getattr(e, "status_code", None) not in RETRY_STATUSES:
                raise
            time.sleep(delay)
