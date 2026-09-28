"""Инференс модели: Qwen2.5-7B-Instruct, локально, 4 бита (bitsandbytes).

При смене модели переписывается только этот файл. Он должен предоставить
    chat(messages, tools) -> Reply
Формат messages, tools и Reply описан в agent.py (раздел «Контракт с моделью»).
"""
import json
import re
import uuid
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from agent import Reply, ToolCall

MODEL_DIR = Path(__file__).parent / "models" / "Qwen2.5-7B-Instruct" / "weights"
MAX_ANSWER_TOKENS = 512
TEMPERATURE = 0.2  # низкая — модель точнее пересказывает данные из инструментов
# Маленькая модель часто отвечает «из головы», не вызывая инструменты.
# Поэтому на каждый вопрос клиента её ответ начинается с <tool_call>:
# сначала она обязана заглянуть в базу, и только потом отвечать.
FORCE_TOOL_CALL_FIRST = True
TOOL_CALL_PREFIX = "<tool_call>\n"

# Qwen2.5 вызывает инструменты так: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

print("Загрузка модели...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    quantization_config=bnb_config,
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)


def chat(messages: list[dict], tools: list[dict]) -> Reply:
    # История в формате agent.py подходит шаблону Qwen2.5 без преобразований:
    # он сам вставляет описания tools в системный промпт, вызовы — в <tool_call>,
    # результаты — в <tool_response>.
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    # последнее сообщение от клиента — значит, это первый шаг ответа: заставляем вызвать инструмент
    forced = FORCE_TOOL_CALL_FIRST and bool(tools) and messages[-1]["role"] == "user"
    if forced:
        text += TOOL_CALL_PREFIX
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_ANSWER_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
        )
    new_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
    # <tool_call> — не служебный токен, skip_special_tokens его не удаляет
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    if forced:  # префикс был частью промпта, а не ответа — возвращаем его, чтобы вызов разобрался
        response = TOOL_CALL_PREFIX + response
    return parse_response(response)


def parse_response(response: str) -> Reply:
    """Текст модели -> Reply: вызовы из <tool_call> в tool_calls, остальное в content."""
    calls = []
    for raw in TOOL_CALL_RE.findall(response):
        try:
            data = json.loads(raw)
            name, args = data["name"], data.get("arguments") or {}
            if isinstance(args, str):  # модель иногда кладёт аргументы строкой
                args = json.loads(args)
        except (json.JSONDecodeError, KeyError, TypeError):
            # отдаём агенту как неизвестный вызов: он вернёт модели ошибку, и она попробует снова
            name, args = "invalid_tool_call", {"raw": raw}
        calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args))

    content = TOOL_CALL_RE.sub("", response).strip()
    if "<tool_call>" in content:  # вызов оборвался по лимиту токенов
        content = content.split("<tool_call>")[0].strip()
        calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name="invalid_tool_call", arguments={}))
    return Reply(content=content, tool_calls=calls)
