#!/usr/bin/env python3
"""CLI: seed / refresh coding-ai worker records in worker_management.

Bulk operations:

    uv run python scripts/register_workers.py            # create missing only
    uv run python scripts/register_workers.py --dry-run  # show what would change
    uv run python scripts/register_workers.py --force    # overwrite every record

Source of truth: ``coding-ai/src/config/agents.py`` (the 14 agent records
extracted from ``src/services/ai_*.py`` ``make_agent_runner`` call sites).

For per-startup self-healing (recreate missing workers when WM's database
is flushed or coding-ai boots into a new environment), see
``src/wm_bootstrap.ensure_workers_registered`` invoked from the FastAPI
lifespan in ``src/main.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import httpx

# Make `src/` importable when invoked as `python scripts/register_workers.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wm_bootstrap import ensure_workers_registered  # noqa: E402

logger = logging.getLogger("register_workers")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Print actions without calling WM.")
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing records via PUT. Without this flag, existing "
            "records are left untouched (operator edits preserved)."
        ),
    )
    p.add_argument(
        "--wm-url",
        default=os.getenv("WORKER_MANAGEMENT_URL"),
        help="worker_management base URL (defaults to $WORKER_MANAGEMENT_URL).",
    )
    p.add_argument(
        "--license-token",
        default=os.getenv("CODING_AI_LICENSE_TOKEN"),
        help=(
            "Service license token used as Bearer in X-Tenant-Id "
            "(defaults to $CODING_AI_LICENSE_TOKEN)."
        ),
    )
    p.add_argument(
        "--inference-router-url",
        default=os.getenv(
            "INFERENCE_ROUTER_URL",
            "http://inference-router:8150/api",
        ),
        help=(
            "URL stored as base_url_override on each worker record. Ops can "
            "PUT a different value per worker after seeding."
        ),
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    if not args.wm_url:
        logger.error("WORKER_MANAGEMENT_URL is required (or pass --wm-url).")
        return 2
    if not args.license_token:
        logger.error("CODING_AI_LICENSE_TOKEN is required (or pass --license-token).")
        return 2

    async with httpx.AsyncClient() as http:
        actions = await ensure_workers_registered(
            http,
            wm_base_url=args.wm_url,
            license_token=args.license_token,
            base_url_override=args.inference_router_url,
            overwrite=args.force,
            dry_run=args.dry_run,
        )

    counts = Counter(actions.values())
    for name, action in sorted(actions.items()):
        print(f"{action:>16}  {name}")
    print()
    print(f"summary: {dict(counts)} (total {len(actions)})")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
