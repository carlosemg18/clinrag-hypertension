"""Model backends for citation-enforced generation.

Claude and Gemini are both constrained to return the same `LLMAnswer` schema —
Claude via forced tool use, Gemini via a JSON response schema — so the two
models can be compared on identical terms. Temperature is pinned to 0.
"""

from __future__ import annotations

from typing import Callable

from clinrag.config import SETTINGS
from clinrag.schema import LLMAnswer

# Claude is not run with extended thinking, so a tight cap is fine.
CLAUDE_MAX_TOKENS = 2048

# gemini-2.5-pro is a thinking model: reasoning tokens are drawn from the same
# output budget, so the budget must cover thinking AND the JSON answer.
GEMINI_THINKING_BUDGET = 4096
GEMINI_MAX_TOKENS = 12288

# --- Claude (Anthropic SDK) -------------------------------------------------

_claude_client = None
_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the grounded, citation-backed answer in the required schema.",
    "input_schema": LLMAnswer.model_json_schema(),
}


def _claude():
    global _claude_client
    if _claude_client is None:
        import anthropic

        if not SETTINGS.anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set (see .env.example).")
        _claude_client = anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
    return _claude_client


def call_claude(system: str, user_message: str) -> LLMAnswer:
    response = _claude().messages.create(
        model=SETTINGS.claude_model,
        max_tokens=CLAUDE_MAX_TOKENS,
        temperature=0,
        system=system,
        tools=[_ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{"role": "user", "content": user_message}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return LLMAnswer.model_validate(block.input)
    raise RuntimeError("Claude returned no tool_use block")


# --- Gemini (google-genai SDK) ----------------------------------------------

_gemini_client = None


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        if not SETTINGS.google_api_key:
            raise SystemExit("GOOGLE_API_KEY is not set (see .env.example).")
        _gemini_client = genai.Client(api_key=SETTINGS.google_api_key)
    return _gemini_client


def call_gemini(system: str, user_message: str) -> LLMAnswer:
    from google.genai import types

    response = _gemini().models.generate_content(
        model=SETTINGS.gemini_model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_schema=LLMAnswer,
            thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
            max_output_tokens=GEMINI_MAX_TOKENS,
        ),
    )
    if isinstance(response.parsed, LLMAnswer):
        return response.parsed
    if response.text:
        return LLMAnswer.model_validate_json(response.text)
    reason = (
        response.candidates[0].finish_reason if response.candidates else "no candidates"
    )
    raise RuntimeError(f"Gemini returned no parseable output (finish_reason={reason})")


# --- Dispatch ---------------------------------------------------------------

BACKENDS: dict[str, Callable[[str, str], LLMAnswer]] = {
    "claude": call_claude,
    "gemini": call_gemini,
}


def generate_answer(model: str, system: str, user_message: str) -> LLMAnswer:
    if model not in BACKENDS:
        raise ValueError(f"Unknown model '{model}'. Options: {sorted(BACKENDS)}")
    return BACKENDS[model](system, user_message)
