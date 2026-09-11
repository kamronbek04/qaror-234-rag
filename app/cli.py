"""Command line entry point: ``python -m app.cli <command>``."""

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence

from app.core.config import Settings, load_settings
from app.core.container import build_embedder
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.ingestion.pipeline import build_index


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Uzbek letters on Windows consoles
    args = _parser().parse_args(argv)
    try:
        settings = load_settings()
        configure_logging(settings.log_level)
        handler: Callable[[Settings, argparse.Namespace], int] = args.handler
        return handler(settings, args)
    except AppError as exc:
        print(f"Xatolik ({exc.code}): {exc.message}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description="234-son qaror bo'yicha RAG API buyruqlari"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="hujjatni indekslash")
    ingest.add_argument("--refresh", action="store_true", help="lex.uz'dan qayta yuklash")
    ingest.add_argument("--strategy", choices=["structural", "fixed"], help="chunking usuli")
    ingest.add_argument(
        "--force", action="store_true", help="indeks dolzarb bo'lsa ham qayta qurish"
    )
    ingest.set_defaults(handler=_ingest)
    return parser


def _ingest(settings: Settings, args: argparse.Namespace) -> int:
    if args.strategy:
        settings = settings.model_copy(update={"chunk_strategy": args.strategy})
    manifest = asyncio.run(
        build_index(
            settings, embedder=build_embedder(settings), refresh=args.refresh, force=args.force
        )
    )
    print(
        f"Indeks tayyor: {manifest.chunk_count} ta bo'lak, model {manifest.embed_model}, "
        f"fingerprint {manifest.fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
