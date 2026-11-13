"""Shared type constraints for coding-ai."""

from __future__ import annotations

SAFE_STRING_PATTERN = r"^[^\x00-\x08\x0B\x0C\x0E-\x1F\x7F]*$"
