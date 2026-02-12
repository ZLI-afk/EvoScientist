"""LLM model configuration based on LangChain init_chat_model.

This module provides a unified interface for creating chat model instances
with support for multiple providers (Anthropic, OpenAI) and convenient
short names for common models.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model

_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# Model registry: short_name -> (model_id, provider)
MODELS: dict[str, tuple[str, str]] = {
    # Anthropic (ordered by capability)
    "claude-opus-4-6": ("claude-opus-4-6", "anthropic"),
    "claude-opus-4-5": ("claude-opus-4-5-20251101", "anthropic"),
    "claude-sonnet-4-5": ("claude-sonnet-4-5-20250929", "anthropic"),
    "claude-haiku-4-5": ("claude-haiku-4-5-20251001", "anthropic"),
    # OpenAI
    "gpt-5.2-codex": ("gpt-5.2-codex", "openai"),
    "gpt-5.2": ("gpt-5.2-2025-12-11", "openai"),
    "gpt-5.1": ("gpt-5.1-2025-11-13", "openai"),
    "gpt-5": ("gpt-5-2025-08-07", "openai"),
    "gpt-5-mini": ("gpt-5-mini-2025-08-07", "openai"),
    "gpt-5-nano": ("gpt-5-nano-2025-08-07", "openai"),
    # Google GenAI
    "gemini-3-pro": ("gemini-3-pro-preview", "google-genai"),
    "gemini-3-flash": ("gemini-3-flash-preview", "google-genai"),
    "gemini-2.5-flash": ("gemini-2.5-flash", "google-genai"),
    "gemini-2.5-flash-lite": ("gemini-2.5-flash-lite", "google-genai"),
    "gemini-2.5-pro": ("gemini-2.5-pro", "google-genai"),
    # NVIDIA
    "glm4.7": ("z-ai/glm4.7", "nvidia"),
    "deepseek-v3.2": ("deepseek-ai/deepseek-v3.2", "nvidia"),
    "deepseek-v3.1": ("deepseek-ai/deepseek-v3.1-terminus", "nvidia"),
    "kimi-k2.5": ("moonshotai/kimi-k2.5", "nvidia"),
    "kimi-k2-thinking": ("moonshotai/kimi-k2-thinking", "nvidia"),
    "minimax-m2.1": ("minimaxai/minimax-m2.1", "nvidia"),
    "step-3.5-flash": ("stepfun-ai/step-3.5-flash", "nvidia"),
    "nemotron-nano": ("nvidia/nemotron-3-nano-30b-a3b", "nvidia"),
    # SiliconFlow (OpenAI-compatible API)
    "glm-5": ("Pro/zai-org/GLM-5", "siliconflow"),
}

DEFAULT_MODEL = "claude-sonnet-4-5"


def get_chat_model(
    model: str | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> Any:
    """Get a chat model instance.

    Args:
        model: Model name (short name like 'claude-sonnet-4-5' or full ID
               like 'claude-sonnet-4-5-20250929'). Defaults to DEFAULT_MODEL.
        provider: Override the provider (e.g., 'anthropic', 'openai').
                  If not specified, inferred from model name or defaults to 'anthropic'.
        **kwargs: Additional arguments passed to init_chat_model (e.g., temperature).

    Returns:
        A LangChain chat model instance.

    Examples:
        >>> model = get_chat_model()  # Uses default (claude-sonnet-4-5)
        >>> model = get_chat_model("claude-opus-4-5")  # Use short name
        >>> model = get_chat_model("gpt-4o")  # OpenAI model
        >>> model = get_chat_model("claude-3-opus-20240229", provider="anthropic")  # Full ID
    """
    model = model or DEFAULT_MODEL

    # Look up short name in registry
    if model in MODELS:
        model_id, default_provider = MODELS[model]
        provider = provider or default_provider
    else:
        # Assume it's a full model ID
        model_id = model
        # Try to infer provider from model ID prefix
        if provider is None:
            if model_id.startswith(("claude-", "anthropic")):
                provider = "anthropic"
            elif model_id.startswith(("gpt-", "o1", "davinci", "text-")):
                provider = "openai"
            elif model_id.startswith("gemini"):
                provider = "google-genai"
            elif "/" in model_id:
                provider = "nvidia"
            else:
                provider = "anthropic"  # Default fallback

    # SiliconFlow → route through OpenAI provider with base_url
    _is_siliconflow = provider == "siliconflow"
    if _is_siliconflow:
        kwargs["base_url"] = _SILICONFLOW_BASE_URL
        api_key = os.environ.get("SILICONFLOW_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key
        provider = "openai"

    # Auto-enable thinking for Anthropic models
    if provider == "anthropic" and "thinking" not in kwargs:
        if model_id.startswith("claude-opus-4-6"):
            kwargs["thinking"] = {"type": "adaptive"}
        else:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}

    # Auto-enable reasoning for OpenAI models (not for SiliconFlow)
    if provider == "openai" and not _is_siliconflow and "reasoning" not in kwargs:
        kwargs["reasoning"] = {"effort": "medium", "summary": "auto"}

    # Auto-enable thinking visibility for Google GenAI models
    if provider == "google-genai":
        kwargs.setdefault("include_thoughts", True)

    return init_chat_model(model=model_id, model_provider=provider, **kwargs)


def list_models() -> list[str]:
    """List all available model short names.

    Returns:
        List of model short names that can be passed to get_chat_model().
    """
    return list(MODELS.keys())


def get_model_info(model: str) -> tuple[str, str] | None:
    """Get the (model_id, provider) tuple for a short name.

    Args:
        model: Short model name.

    Returns:
        Tuple of (model_id, provider) or None if not found.
    """
    return MODELS.get(model)
