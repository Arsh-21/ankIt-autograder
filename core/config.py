"""Loads config.yaml and .env, and answers "which model does role X use?"."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# roles that send images, so they need a vision model
VISION_ROLES = {"ocr", "recheck", "grader_vision", "locator"}

_config: dict | None = None


def load_config() -> dict:
    """Load config.yaml once and return the cached copy after that.

    Returns:
        dict: The parsed config.
    """
    global _config
    if _config is None:
        with open(ROOT / "config.yaml", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def list_profiles() -> list[str]:
    """List the profile names in config.yaml.

    Returns:
        list[str]: Profile names in file order.
    """
    return list(load_config()["profiles"].keys())


def resolve_profile_name(name: str | None = None) -> str:
    """Pick the profile to use.

    Args:
        name (str | None): Requested profile; None falls back to the
            AUTOGRADER_PROFILE environment variable, then active_profile.

    Returns:
        str: A profile name that exists in config.yaml.

    Raises:
        ValueError: If the profile is not in config.yaml.
    """
    cfg = load_config()
    name = name or os.getenv("AUTOGRADER_PROFILE") or cfg["active_profile"]
    if name not in cfg["profiles"]:
        raise ValueError(f"Unknown profile '{name}'. Options: {list_profiles()}")
    return name


def get_profile(name: str | None = None) -> dict:
    """Get a profile's settings.

    Args:
        name (str | None): Profile name; None uses the active one.

    Returns:
        dict: The profile block ("vision", "text", optional per-role entries, "max_parallel").
    """
    return load_config()["profiles"][resolve_profile_name(name)]


def model_for(role: str, profile_name: str | None = None) -> dict:
    """Get the model spec for an agent role.

    A role listed in the profile uses its own entry; otherwise image roles
    (VISION_ROLES) use "vision" and the rest use "text".

    Args:
        role (str): Agent role, e.g. "grader" or "ocr".
        profile_name (str | None): Profile name; None uses the active one.

    Returns:
        dict: {"provider", "model", optional "extra_body"}.
    """
    prof = get_profile(profile_name)
    if role in prof:
        return prof[role]
    return prof["vision"] if role in VISION_ROLES else prof["text"]


def max_parallel(profile_name: str | None = None) -> int:
    """Get how many model calls a profile may run at once.

    Args:
        profile_name (str | None): Profile name; None uses the active one.

    Returns:
        int: The profile's max_parallel, 1 if not set.
    """
    return int(get_profile(profile_name).get("max_parallel", 1))


def thresholds() -> dict:
    """Get the `thresholds:` block of config.yaml.

    Returns:
        dict: Threshold name -> value.
    """
    return load_config()["thresholds"]


def price_for(model: str) -> tuple[float, float] | None:
    """Get a model's price from the `prices:` block of config.yaml.

    Args:
        model (str): Model id, e.g. "openai/gpt-4o-mini".

    Returns:
        tuple[float, float] | None: (USD per million input tokens, USD per million
            output tokens), or None if the model has no price entry.
    """
    p = (load_config().get("prices") or {}).get(model)
    return (p["in"], p["out"]) if p else None
