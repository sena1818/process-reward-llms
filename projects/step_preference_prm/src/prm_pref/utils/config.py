from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return parse_simple_yaml(path)

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small subset of YAML used by this project config.

    This supports nested dictionaries, simple lists, booleans, ints, floats,
    and strings. It is intentionally small so the data-audit scripts can run
    before any Python dependencies are installed.
    """

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines: list[tuple[int, str]] = []
    for raw_line in raw_lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    if not lines:
        return {}

    parsed, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError(f"Could not parse full config file: {path}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    return parsed


def _parse_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    current_indent, current_text = lines[index]
    if current_indent != indent:
        raise ValueError(f"Unexpected indentation near: {current_text}")

    if current_text.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_dict(lines, index, indent)


def _parse_dict(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}

    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected nested line without parent key: {text}")
        if text.startswith("- "):
            break
        if ":" not in text:
            raise ValueError(f"Expected key/value line, got: {text}")

        key, value_text = text.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        index += 1

        if value_text:
            result[key] = _parse_scalar(value_text)
            continue

        if index >= len(lines) or lines[index][0] <= current_indent:
            result[key] = {}
            continue

        child_indent = lines[index][0]
        value, index = _parse_block(lines, index, child_indent)
        result[key] = value

    return result, index


def _parse_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    result: list[Any] = []

    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected nested list line: {text}")
        if not text.startswith("- "):
            break

        item_text = text[2:].strip()
        index += 1
        if item_text:
            result.append(_parse_scalar(item_text))
            continue

        if index >= len(lines) or lines[index][0] <= current_indent:
            result.append(None)
            continue

        child_indent = lines[index][0]
        value, index = _parse_block(lines, index, child_indent)
        result.append(value)

    return result, index


def _parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None

    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text
