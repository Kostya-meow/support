from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None
_BOT_RESPONSES_CACHE: dict[str, Any] | None = None
_SIMULATOR_PROMPTS_CACHE: dict[str, Any] | None = None
_TELEGRAM_RESPONSES_CACHE: dict[str, Any] | None = None
_VK_RESPONSES_CACHE: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration once and cache it."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = Path(path) if path else Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _CONFIG_CACHE = yaml.safe_load(fp) or {}
    return _CONFIG_CACHE


def load_bot_responses(path: str | Path | None = None) -> dict[str, Any]:
    """Load bot responses configuration once and cache it."""
    global _BOT_RESPONSES_CACHE
    if _BOT_RESPONSES_CACHE is not None:
        return _BOT_RESPONSES_CACHE

    config_path = Path(path) if path else Path("bot_responses.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Bot responses config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _BOT_RESPONSES_CACHE = yaml.safe_load(fp) or {}
    return _BOT_RESPONSES_CACHE


def load_telegram_responses(path: str | Path | None = None) -> dict[str, Any]:
    """Load Telegram responses configuration once and cache it."""
    global _TELEGRAM_RESPONSES_CACHE
    if _TELEGRAM_RESPONSES_CACHE is not None:
        return _TELEGRAM_RESPONSES_CACHE

    config_path = Path(path) if path else Path("telegram_responses.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Telegram responses config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _TELEGRAM_RESPONSES_CACHE = yaml.safe_load(fp) or {}
    return _TELEGRAM_RESPONSES_CACHE


def load_vk_responses(path: str | Path | None = None) -> dict[str, Any]:
    """Load VK responses configuration once and cache it."""
    global _VK_RESPONSES_CACHE
    if _VK_RESPONSES_CACHE is not None:
        return _VK_RESPONSES_CACHE

    config_path = Path(path) if path else Path("vk_responses.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"VK responses config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _VK_RESPONSES_CACHE = yaml.safe_load(fp) or {}
    return _VK_RESPONSES_CACHE


def load_simulator_prompts(path: str | Path | None = None) -> dict[str, Any]:
    """Load simulator prompts configuration once and cache it."""
    global _SIMULATOR_PROMPTS_CACHE
    if _SIMULATOR_PROMPTS_CACHE is not None:
        return _SIMULATOR_PROMPTS_CACHE

    config_path = Path(path) if path else Path("simulator_prompts.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Simulator prompts config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        _SIMULATOR_PROMPTS_CACHE = yaml.safe_load(fp) or {}
    return _SIMULATOR_PROMPTS_CACHE


def reset_cache() -> None:
    global _CONFIG_CACHE, _BOT_RESPONSES_CACHE, _SIMULATOR_PROMPTS_CACHE, _TELEGRAM_RESPONSES_CACHE, _VK_RESPONSES_CACHE
    _CONFIG_CACHE = None
    _BOT_RESPONSES_CACHE = None
    _SIMULATOR_PROMPTS_CACHE = None
    _TELEGRAM_RESPONSES_CACHE = None
    _VK_RESPONSES_CACHE = None
