"""Thin compatibility entry point for ``python -m memescanner.main``.

This module holds no scanning logic of its own. It exists so that the legacy
command ``python -m memescanner.main`` keeps working, and it delegates directly
to the unified, evidence-gated runtime in :mod:`memescanner.__main__`.
"""

import asyncio

from memescanner.config import Config


async def main() -> None:
    """Compatibility command redirected to the unified safe default runtime."""
    from memescanner.__main__ import main_loop

    await main_loop(Config.from_env())


if __name__ == "__main__":
    asyncio.run(main())
