"""
LLM client for OpenAI and Anthropic.
prompt can be a string or a list of {"role": ..., "content": ...} messages.
"""

import os
import config


def call_llm(prompt) -> str:
    provider = config.LLM_PROVIDER
    if provider == "anthropic":
        return _call_anthropic(prompt)
    elif provider == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _to_messages(prompt) -> list:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _call_anthropic(prompt) -> str:
    import anthropic
    messages = _to_messages(prompt)
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user_msgs = [m for m in messages if m["role"] != "system"]
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs = dict(model=config.LLM_MODEL, max_tokens=256, messages=user_msgs)
    if system:
        kwargs["system"] = system
    message = client.messages.create(**kwargs)
    return message.content[0].text.strip()


def _call_openai(prompt) -> str:
    import time
    from openai import OpenAI, BadRequestError, RateLimitError
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kwargs = dict(model=config.LLM_MODEL, messages=_to_messages(prompt))
    for attempt in range(8):
        try:
            response = client.chat.completions.create(**kwargs, max_tokens=256, temperature=0.0)
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait = min(60 * (attempt + 1), 300)
            print(f"      [rate limit] waiting {wait}s... (attempt {attempt+1}/8)")
            time.sleep(wait)
        except BadRequestError:
            response = client.chat.completions.create(**kwargs, max_completion_tokens=4096)
            return response.choices[0].message.content.strip()
    raise RuntimeError("OpenAI rate limit: failed after 8 attempts.")
